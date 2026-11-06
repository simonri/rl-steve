"""Real game check with a second, undriven client as the player. Blocks get reached by moving it with commands."""
import sys
import threading
import time
from pathlib import Path

import numpy as np

import demo_data
import demo_server
from auto_env import AutoEnv
from skill_env import Y

port, out = int(sys.argv[1]), Path(sys.argv[2])
# Trimming first, without the game.
base = dict(pos=[0.0, 0.0, 0.0], yaw=0.0, pitch=0.0, forward=False, back=False, left=False, right=False, jump=False, sneak=False)
row = lambda x=0.0, **k: dict(base, pos=[x, 0.0, 0.0], **k)
walk = lambda x: row(x, forward=True)
assert demo_data.trim([row() for _ in range(30)] + [walk(1.0), walk(2.0)]) == [walk(1.0), walk(2.0)], "waiting at the start goes"
assert demo_data.trim([row(yaw=float(i)) for i in range(30)] + [walk(1.0)]) == [walk(1.0)], "looking around on the spot goes too"
short = [walk(1.0)] + [row(1.0) for _ in range(9)] + [walk(2.0)]
assert len(demo_data.trim(short)) == 11, "a short pause stays"
long = [walk(1.0)] + [row(1.0) for _ in range(10)] + [walk(2.0)]
assert demo_data.trim(long) == [walk(1.0), walk(2.0)], "half a second or more goes"
turning = [walk(1.0)] + [row(1.0, yaw=float(i)) for i in range(1, 20)] + [walk(2.0)]
assert len(demo_data.trim(turning)) == 21, "turning is not waiting"
died = [walk(1.0), dict(walk(2.0), event="died"), walk(0.5), walk(1.5)]
assert demo_data.trim(died) == [walk(0.5), walk(1.5)], "only the last try"
print("trim ok", flush=True)
server = demo_server.Server(port, out)
c = server.c
for _ in range(120):
    reply = c.call(op="human_ticks")
    if reply.get("player"):
        break
    time.sleep(1)
player = reply.get("player")
assert player, "the test player never joined"
print("player:", player, flush=True)
folder = out / "test"
real_update = demo_server.Round.update

# Only rounds that were won are saved.
script = {"mode": "fall"}
lock = threading.Lock()
real_call = c.call
c.call = lambda **msg: locked(real_call, msg)


def locked(call, msg):
    with lock:  # the helper below shares this connection
        return call(**msg)


def follow():
    """Plays the role of the person's movement: the server moves them, and the normal round logic sees it."""
    while not script.get("stop"):
        time.sleep(0.6)
        r = script.get("round")
        if r is None:
            continue
        if script["mode"] == "fall":
            pos = [0.5, Y - 6.0, 0.5]
        else:
            x, y, z = r.goal_cell
            pos = [x + 0.5, float(y), z + 0.5]
        c.call(op="human_move", pos=pos, yaw=0.0, pitch=0.0)


real_draw = server.draw


def draw_and_note():
    script["round"] = real_draw()
    return script["round"]


server.draw = draw_and_note
thread = threading.Thread(target=follow, daemon=True)
thread.start()
folder.mkdir(parents=True, exist_ok=True)
server.play_round(player, folder)
assert not list(folder.glob("*.gz")), "a lost round must not be saved"
print("lost round not saved", flush=True)
script["round"], script["mode"] = None, "win"
server.force = ("path", dict(length=4))
server.play_round(player, folder)
won_round = script["round"]
script["round"] = None
saved = sorted(folder.glob("*.gz"))
assert len(saved) == 1
head, ticks, end = demo_data.load(saved[0])
print("won round:", end, head["meta"]["family"], flush=True)
assert end["result"] == "won" and end["kept"] == len(ticks) and ticks[-1]["event"] == "won"
goal = won_round.goal
assert c.call(op="human_block", at=list(goal))["name"] == "minecraft:redstone_block", goal
if head["meta"]["family"] == "path":
    assert [goal[0], goal[2]] == head["meta"]["goal"] and goal == tuple(head["meta"]["course"][-1])
assert end["kept"] < end["ticks"], "the waiting is cut out"
assert not any(t.get("event") == "died" for t in ticks)
script["stop"] = True
thread.join(timeout=3)
# A bridge round: blocks in the hand, and the goal is the middle of the far platform.
server.force = ("bridge", dict(gap=5))
script["stop"] = False
script["mode"], script["round"] = "win", None
thread = threading.Thread(target=follow, daemon=True)
thread.start()
server.play_round(player, folder)
bridge_round = script["round"]
script["round"], script["stop"] = None, True
thread.join(timeout=3)
files = sorted(folder.glob("*.gz"))
assert len(files) == 2, files
head, bridge_ticks, end = demo_data.load(files[-1])
print("bridge round:", end, "blocks", head["meta"]["blocks_given"], "hit seen", any("hit" in t for t in bridge_ticks), flush=True)
assert head["meta"]["family"] == "bridge" and head["meta"]["blocks_given"] > 0
assert bridge_ticks[-1]["event"] == "won" and {t["held_count"] for t in bridge_ticks} == {head["meta"]["blocks_given"]}
assert c.call(op="human_block", at=list(bridge_round.goal))["name"] == "minecraft:redstone_block"
b_obs, b_act, _ = demo_data.replay(files[-1])
assert b_obs.shape[1] == AutoEnv.N_OBS and np.isfinite(b_obs).all()
# The real map: its blocks are in the world, the goal is on the island, and it all goes away afterward.
server.force = ("treenan", dict(gap=10))
script["stop"], script["mode"], script["round"] = False, "win", None
thread = threading.Thread(target=follow, daemon=True)
thread.start()
server.play_round(player, folder)
map_round = script["round"]
script["round"], script["stop"] = None, True
thread.join(timeout=3)
files = sorted(folder.glob("*.gz"), key=lambda p: p.stat().st_mtime)
head, map_ticks, end = demo_data.load(files[-1])
print("treenan round:", end, "blocks", head["meta"]["blocks_given"], flush=True)
assert head["meta"]["family"] == "treenan" and end["result"] == "won"
assert c.call(op="human_block", at=list(map_round.goal))["name"] == "minecraft:redstone_block"
some = next(iter(map_round.env.next_cells))
assert c.call(op="human_block", at=list(some))["name"] != "minecraft:air", some
m_obs, m_act, _ = demo_data.replay(files[-1])
assert m_obs.shape[1] == AutoEnv.N_OBS and np.isfinite(m_obs).all()
server.force = ("path", dict(length=4))
script["stop"], script["round"] = False, None
thread = threading.Thread(target=follow, daemon=True)
thread.start()
server.play_round(player, folder)
script["stop"] = True
thread.join(timeout=3)
assert c.call(op="human_block", at=list(some))["name"] == "minecraft:air", "the map is gone"
# Rounds are drawn at random over every kind, with no difficulty that climbs.
server.force = None
kinds = {server.draw().family for _ in range(40)}
assert kinds == set(demo_server.PLAYED), kinds
# What the model would see and press, per tick, in the training format.
obs, act, won = demo_data.replay(saved[0])
assert obs.shape[1] == AutoEnv.N_OBS and act.shape[1] == 8 and np.isfinite(obs).all() and len(obs) == len(act)  # a teleported test player presses nothing, so little is left
assert (act.min(0) >= 0).all() and (act.max(0) < [3, 3, 2, 2, 2, 2, 11, 11]).all()
print(saved[0].name, obs.shape, "clean" if won else "after a death", flush=True)
print("keys in the recording:", sorted(k for k in ticks[0] if k not in ("solid", "feet")), flush=True)
c.call(op="human_record", on=False)
print("ok")
