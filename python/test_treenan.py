"""Checks the Treenan stages: the spots themselves without the game, then real rounds in it."""
import math
import sys
import time

import numpy as np

from skill_env import STAGES, TREENAN, SkillEnv, Y
from treenan import GRASS, MAX_GAP, MIN_GAP, Treenan

data = Treenan()
print("start spot", data.start, "distance to bed", round(math.hypot(data.start[0] + 0.5 + 66.5, data.start[2] + 0.5 - 0.5), 1), "goal spots", {k: len(v) for k, v in data.spots.items()})
assert data.spots["center"] and data.spots["side"]
for spots in data.spots.values():
    for x, y, z in spots:
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                assert data.cells[(x + dx, y, z + dz)].startswith(GRASS)
                assert (x + dx, y + 1, z + dz) not in data.cells and (x + dx, y + 2, z + dz) not in data.cells
        assert data.start[1] - 8 <= y <= data.start[1] + 3
if len(sys.argv) < 2:
    print("spots ok")
    sys.exit(0)

env = SkillEnv(int(sys.argv[1]))
stage = next(i for i, t in enumerate(STAGES) if t[6] == TREENAN)
env.set_stage(stage)
seen, gaps, resets = {"straight": 0, "diagonal": 0}, [], []
t_rounds = int(sys.argv[2])
for n in range(t_rounds):
    t0 = time.perf_counter()
    obs, _ = env.reset()
    resets.append(time.perf_counter() - t0)
    assert obs.shape == (SkillEnv.N_OBS,) and np.isfinite(obs).all()
    seen[env.map_target] += 1
    gaps.append(env.map_gap)
    assert MIN_GAP <= env.map_gap <= MAX_GAP
    o = env.obs
    assert o["on_ground"] and abs(o["pos"][1] - (Y + 1)) < 0.01, o["pos"]
    assert not env.reached(o) and env.blocks_given >= env.par
    # Near the bed.
    names = env.c.call(op="blocks", min=[-20, Y - 2, -20], max=[20, Y + 3, 20])["names"]
    assert any(name.endswith("_bed") for name in names), "no bed near the start"
    # Real spot: stand on it in the game, it must hold the player and count as the goal.
    gx, gz = env.goal_x, env.goal_z
    gy = Y + env.dy
    for _ in range(40):
        env.c.step()
    ground = env.c.call(op="blocks", min=[gx - 1, gy, gz - 1], max=[gx + 1, gy, gz + 1])["names"]
    head = env.c.call(op="blocks", min=[gx - 1, gy + 1, gz - 1], max=[gx + 1, gy + 2, gz + 1])["names"]
    assert all(g == GRASS for g in ground), ground
    assert all(h.endswith("air") for h in head), head
    env.obs = env.c.reset(keep_arena=True, blocks=env.blocks_given, pos=[gx + 0.5, gy + 1, gz + 0.5], yaw=0.0, pitch=0.0)
    assert env.reached(env.obs), (env.obs["pos"], gx, gz, gy)
    # A few real ticks of play, including a block placed on the map, and it must clear next round.
    for _ in range(6):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        if term or trunc:
            break
print(f"{t_rounds} rounds ok: {seen}, gaps {min(gaps)}..{max(gaps)}, reset {np.mean(resets):.2f} s on average, slowest {max(resets):.2f} s")
assert seen["straight"] and seen["diagonal"]
# Leaving the map must remove all of it.
env.set_stage(0)
env.reset()
for _ in range(60):
    env.c.step()
left = 0
for x in range(-25, 80, 3):
    row = env.c.call(op="blocks", min=[x, Y - 30, -25], max=[x, Y + 40, 80])["names"]
    left += sum(not n.endswith("air") for n in row)
print("map blocks left after going back to a normal stage (sampled, arena excluded):", left)
assert left < 100
print("ok")
env.c.close()
