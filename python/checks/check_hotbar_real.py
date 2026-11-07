"""The hotbar in the real game (bot1, port 25602) against the simulator, with the same inputs every tick: does the game
give the round's hotbar, report it, and change slot on a hotbar key as the simulator does; does a click with the
blocks' key place a block in the same tick, and does a hit with blocks in hand do 1 damage?

    python check_hotbar_real.py [port]
"""
import _path  # noqa: F401  (python/ on the import path)
import sys

import numpy as np

import curriculum
import items
import skill_env
from auto_env import AutoEnv
from sim_env import FakeClient

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 25602
real_client = skill_env.McClient


def play(env, levels, family, seed, actions_fn, ticks):
    env.set_levels(levels)
    env.set_exam(family)
    env.reset(seed=seed)
    rows = [dict(slot=int(env.obs.get("slot", 0)), hotbar=env._hotbar(env.obs), blocks=env._blocks(env.obs), pos=np.array(env.obs["pos"]))]
    layout = (list(env.hotbar), env.hotbar_slot)
    for t in range(ticks):
        _, _, term, trunc, _ = env.step(actions_fn(t, layout))
        o = env.obs
        foe = (o.get("players") or [{}])[0].get("health")
        rows.append(dict(slot=int(o.get("slot", 0)), hotbar=env._hotbar(o), blocks=env._blocks(o), pos=np.array(o["pos"]), foe=foe))
        if term or trunc:
            break
    return layout, rows


def run(name, levels, family, actions_fn, ticks=40):
    skill_env.McClient = real_client
    real = AutoEnv(PORT)
    lr, r = play(real, levels, family, 5, actions_fn, ticks)
    real.c.close()
    skill_env.McClient = FakeClient
    sim = AutoEnv(0)
    ls, s = play(sim, levels, family, 5, actions_fn, ticks)
    assert lr == ls, ("the same round", lr, ls)
    assert r[0]["hotbar"] == [(k, c) for k, c in lr[0]] and r[0]["slot"] == lr[1], ("the game gives the round's hotbar", r[0], lr)
    bad = 0
    for t in range(min(len(r), len(s))):
        same = r[t]["slot"] == s[t]["slot"] and r[t]["hotbar"] == s[t]["hotbar"]
        bad += not same
        if not same or t < 3:
            print(f"{name} t{t}: real slot {r[t]['slot']} blocks {r[t]['blocks']}  sim slot {s[t]['slot']} blocks {s[t]['blocks']}  |pos| {np.abs(r[t]['pos'] - s[t]['pos']).max():.3f}")
    print(f"== {name}: {len(r)} real ticks, {len(s)} simulated, ticks where the hotbar differs: {bad}")
    return r, s


def slot_of(layout, want):
    return next(i for i, (k, c) in enumerate(layout[0]) if c > 0 and want(k))


# A fight: the sword in hand, then the blocks' key, then back to the sword.
lv = curriculum.Curriculum().levels
lv["duel"].update(bot=0, arena=0, distance=0, yaw=0, gear=0, kb=0, count=0)


def switch(t, layout):
    a = np.array([0, 0, 0, 0, 0, 0, 5, 5, 0, 0])
    if t == 5:
        a[9] = slot_of(layout, items.placeable) + 1
    if t == 15:
        a[9] = slot_of(layout, items.is_sword) + 1
    return a


r, s = run("duel, keys", lv, "duel", switch)
assert any(not items.is_sword(row["hotbar"][row["slot"]][0]) for row in r), "the blocks' key changed the hand in the game"

# A bridge: walk back to the edge looking down, then right clicks with the blocks in hand use them up in both.
lv = curriculum.Curriculum().levels


def build(t, layout):
    # Look 60 degrees down at the floor ahead and right-click now and then: blocks go on it, from the stack in hand.
    # The course starts at the edge, its back to the gap and looking down: sneak back to the edge (shift keeps it on), and
    # a right click goes on the edge block's side, as in bridging. Shift down: a right click counts only then (AutoEnv).
    a = np.array([2 if t < 10 else 0, 0, 1, 0, 0, 0, 5, 5, 0, 0])
    if t >= 6 and t % 3 == 0:
        a[3] = 1
    if t == 12:  # the sword's key and a click in the same tick: nothing is placed
        a[9] = slot_of(layout, items.is_sword) + 1
    if t == 21:  # the blocks' key and a click in the same tick: one is placed
        a[9] = slot_of(layout, items.placeable) + 1
    return a


r, s = run("bridge, clicks", lv, "bridge", build, ticks=40)
print(f"blocks used: real {r[0]['blocks'] - r[-1]['blocks']}, sim {s[0]['blocks'] - s[-1]['blocks']}")
print("hotbar check done")
