"""How fast the simulator runs rounds, without a model: decisions per second for different numbers of rounds at once.

    python sim_bench.py 64 256 1024
"""
import sys
import time

import numpy as np

import curriculum
from sim_env import SimVecEnv


def bench(n, seconds=5.0):
    levels = curriculum.Curriculum().levels
    for f in ("hop", "path", "bridge"):
        for d in levels[f]:
            if d not in curriculum.EXPERT.get(f, ()):
                levels[f][d] = len(curriculum.FAMILIES[f][d]) // 2
    env = SimVecEnv(n, levels)
    env.reset()
    rng = np.random.default_rng(0)
    acts = np.stack([rng.integers(0, s, (n,)) for s in env.action_space.nvec], 1)
    steps, resets = 0, 0
    t0 = time.perf_counter()
    in_rust = 0.0
    while time.perf_counter() - t0 < seconds:
        if steps % 8 == 0:
            acts = np.stack([rng.integers(0, s, (n,)) for s in env.action_space.nvec], 1)
        env.step_async(acts)
        a = time.perf_counter()
        env.batch.step(env.actions, env.obs, env.rewards, env.dones)
        in_rust += time.perf_counter() - a
        for i in np.flatnonzero(env.dones):
            env._new(int(i))
            resets += 1
        steps += 1
    took = time.perf_counter() - t0
    print(f"{n} rounds: {steps * n / took:,.0f} decisions/s, {resets / took:,.0f} new rounds/s, stepping {in_rust / took:.0%} of the time, drawing courses the rest", flush=True)


if __name__ == "__main__":
    for n in map(int, sys.argv[1:] or ["64", "256"]):
        bench(n)
