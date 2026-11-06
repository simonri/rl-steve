"""Real game check: every course kind builds and resets at all levels, and a script clears zigzag jumps in the new setup."""
import _path  # noqa: F401  (python/ on the import path)
import math
import sys
import time

import numpy as np

from auto_env import NO_TURN, TURN_STEPS, AutoEnv
from curriculum import FAMILIES, Curriculum

env = AutoEnv(int(sys.argv[1]))
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
turn = lambda e: min(range(len(TURN_STEPS)), key=lambda i: abs(e / 2 - TURN_STEPS[i]))


def standoff():
    """Stands still at about 2.3 to 3 from the next block, faces it with the sprint key down, then jumps."""
    o = env.obs
    x, _, z = o["pos"]
    tx, _, tz = env._next_target()
    d = math.hypot(tx - x, tz - z)
    ahead, side = env._way_in_view(o)
    y = turn(math.degrees(math.atan2(side, ahead)))
    speed = math.hypot(o["vel"][0], o["vel"][2])
    if not o["on_ground"]:
        return [1, 0, 0, 0, 0, 1, y, NO_TURN] if d > 0.5 else [0, 0, 0, 0, 0, 0, y, NO_TURN]
    if ahead < 0.99 or speed > 0.03:
        return [0, 0, 0, 0, 0, 1, y, NO_TURN]
    return [1, 0, 0, 0, 1, 1, y, NO_TURN]


# All levels of all kinds: builds, resets and steps with finite values.
for family, dials in FAMILIES.items():
    env.levels = Curriculum({family: {d: len(v) - 1 for d, v in dials.items()}, "bridge": {"gap": 5, "directions": 2}}).levels
    env.set_exam(family)
    t0 = time.time()
    for _ in range(4):
        obs, _ = env.reset()
        assert np.isfinite(obs).all() and obs.shape == (AutoEnv.N_OBS,)
        for _ in range(20):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(obs).all() and np.isfinite(r)
            if term:
                break
    print(f"{family}: all dials at the limit ok, {(time.time() - t0) / 4:.2f} s per round", flush=True)

# The hardest thing the old stage asked for, at a growing late-key chance.
env.set_exam("hop")
for late in (0.0,):
    levels = Curriculum().levels
    levels["hop"].update(offset=4, takeoff=1, count=3)
    env.levels = levels
    wins, landed = 0, 0
    for _ in range(rounds):
        env.reset()
        done, info = False, {}
        while not done:
            _, _, term, trunc, info = env.step(standoff())
            done = term or trunc
        wins += info["clean"]
        landed += info["landed"]
    print(f"script, zigzag from a block, 4 jumps, late {late}: {wins}/{rounds} clean, {landed / rounds:.2f} landed", flush=True)

# Speed of rounds that die right away, now without time spent dead.
levels = Curriculum().levels
env.levels = levels
t0 = time.time()
n = 0
for _ in range(20):
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = env.step([1, 0, 0, 0, 0, 1, 3, NO_TURN])
        done = term or trunc
        n += 1
print(f"{n / (time.time() - t0):.0f} decisions per second in runs that walk off", flush=True)
env.c.close()
print("ok")
