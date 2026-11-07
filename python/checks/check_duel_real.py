"""The same duel in the real game (bot1, port 25602) and in the simulator, with the same inputs every tick: are the hits,
the damage and the knockback the same? Prints the first ticks side by side and the largest differences.

    python check_duel_real.py [port] [cases, e.g. 0,2]
"""
import _path  # noqa: F401  (python/ on the import path)
import sys

import numpy as np

import curriculum
import skill_env
from auto_env import AutoEnv
from sim_env import FakeClient

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 25602
real_client = skill_env.McClient


def play(env, levels, seed, actions):
    env.set_levels(levels)
    env.set_exam("duel")
    env.reset(seed=seed)
    rows = []
    for a in actions:
        _, _, term, trunc, info = env.step(a)
        o = env.obs
        p = (o.get("players") or [{}])[0]
        rows.append(dict(pos=np.array(o["pos"]), vel=np.array(o["vel"]), health=o["health"], foe_health=p.get("health"), foe_pos=np.array(p.get("pos", [0, 0, 0])),
                         foe_vel=np.array(p.get("vel", [0, 0, 0]))))
        if term or trunc:
            break
    return rows


def case(name, bot, actions_fn, ticks=120):
    levels = curriculum.Curriculum().levels
    levels["duel"].update(bot=bot, arena=0, distance=0, yaw=0, gear=0, kb=0, count=0)
    acts = [actions_fn(t) for t in range(ticks)]
    skill_env.McClient = real_client
    real = AutoEnv(PORT)
    r = play(real, levels, 11, acts)
    real.c.close()
    skill_env.McClient = FakeClient
    sim = AutoEnv(0)
    s = play(sim, levels, 11, acts)
    n = min(len(r), len(s))
    print(f"== {name}: {len(r)} real ticks, {len(s)} simulated")
    print(" t   real health / foe      sim health / foe     |pos diff|  |foe pos diff|")
    worst_pos = worst_foe = 0.0
    health_diff = foe_diff = 0
    for t in range(n):
        dp = float(np.abs(r[t]["pos"] - s[t]["pos"]).max())
        df = float(np.abs(r[t]["foe_pos"] - s[t]["foe_pos"]).max())
        worst_pos, worst_foe = max(worst_pos, dp), max(worst_foe, df)
        health_diff += abs(r[t]["health"] - s[t]["health"]) > 1e-3
        foe_diff += r[t]["foe_health"] is not None and abs(r[t]["foe_health"] - s[t]["foe_health"]) > 1e-3
        if t < 40 or (r[t]["health"] != s[t]["health"]) or (r[t]["foe_health"] != s[t]["foe_health"]):
            if t < 40 or health_diff + foe_diff < 8:
                print(f"{t:3d}  {r[t]['health']:6.2f} / {r[t]['foe_health'] or 0:6.2f}     {s[t]['health']:6.2f} / {s[t]['foe_health'] or 0:6.2f}     {dp:7.3f}  {df:7.3f}")
    print(f"ticks with different own health {health_diff}, different opponent health {foe_diff}, largest position difference {worst_pos:.3f} own, {worst_foe:.3f} opponent")


def hit_dummy(t):
    a = np.array([1 if t < 12 else 0, 0, 0, 0, 0, 0, 5, 5, 1 if t % 3 == 0 else 0])
    return a


def stand(t):
    return np.array([0, 0, 0, 0, 0, 0, 5, 5, 0])


def sprint_hit_dummy(t):
    return np.array([1 if t < 14 else 0, 0, 0, 0, 0, 1 if t < 14 else 0, 5, 5, 1 if t % 3 == 0 else 0])


ONLY = sys.argv[2] if len(sys.argv) > 2 else None
CASES = [
    ("the model walks to the dummy and hits it every third tick", 0, hit_dummy),
    ("the walker hits the model, which stands still", 1, stand),
    ("the model sprints at the dummy and hits it every third tick", 0, sprint_hit_dummy),
    ("the strafer fights the model, which stands still", 2, stand),
    ("the combo bot (w-taps) fights the model, which stands still", 3, stand),
    ("the pusher fights the model, which stands still", 4, stand),
]
for i, (name, bot, fn) in enumerate(CASES):
    if ONLY is None or str(i) in ONLY.split(","):
        case(name, bot, fn)
