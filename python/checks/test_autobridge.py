"""Real game check: a client with the bridging helper plays real bridge rounds on the recording server, straight and diagonal."""
import _path  # noqa: F401  (python/ on the import path)
import sys
import time
from pathlib import Path

import demo_data
import demo_server

port, out, rounds = int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
server = demo_server.Server(port, out)
for _ in range(120):
    player = server.c.call(op="human_ticks").get("player")
    if player:
        break
    time.sleep(1)
assert player, "no player joined"
folder = out / "bridge"
for name, directions, gap in (("straight", "axis", 5), ("straight long", "axis", 8), ("diagonal", "eight", 5), ("diagonal long", "eight", 8), ("any angle", "any", 5)):
    server.force = ("bridge", dict(gap=gap, directions=directions, start="edge", heights=(0,), obstacle=None, yaw=15, budget=(3.0, 2), longer=1.0))
    wins, ticks, kept = 0, [], []
    for _ in range(rounds):
        before = len(list(folder.glob("*.gz"))) if folder.exists() else 0
        real = server.finish_round
        seen = {}
        def note(r, rec, result, real=real, seen=seen):
            seen.update(result=result, ticks=r.ticks, gap=r.env.par, angle=r.env.goal_x, deaths=r.deaths)
            real(r, rec, result)
        server.finish_round = note
        server.play_round(player, folder)
        server.finish_round = real
        wins += seen["result"] == "won" and seen["deaths"] == 0
        ticks.append((seen["result"], seen["ticks"], seen["gap"]))
        time.sleep(0.5)
    good = [t for t in ticks if t[0] == "won"]
    speed = sum(g for _, _, g in good) / max(sum(t for _, t, _ in good) / 20, 0.01)
    print(f"{name}: {wins}/{rounds} clean, about {speed:.2f} blocks/s, {[t[0] for t in ticks]}", flush=True)
files = sorted(folder.glob("*.gz")) if folder.exists() else []
if files:
    head, rows, end = demo_data.load(files[-1])
    placed = rows[0]["held_count"] - rows[-1]["held_count"]
    print("last recording: placed", placed, "of", head["meta"]["blocks_given"], "aim seen in", sum("hit" in t for t in rows), "of", len(rows), "ticks, shift in", sum(t["sneak"] for t in rows), flush=True)
    obs, act, won = demo_data.replay(files[-1])
    print("clicks in the converted data:", int((act[:, 3] > 0).sum()), flush=True)
print("ok")
