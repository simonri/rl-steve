"""The Rust rounds give exactly what AutoEnv gives on the same physics: what the model sees, points, and endings.

AutoEnv runs unchanged on FakeClient (the simulator behind McClient's interface). mcsim.Batch runs the same course
with its own Rust rules. Both get the same inputs; every value must agree.

    python test_sim_env.py [rounds]
"""
import _path  # noqa: F401  (python/ on the import path)
import json
import sys

import numpy as np

import curriculum
import mcsim
import skill_env
from auto_env import AutoEnv
from sim_env import CourseMaker, FakeClient


def reference_env(levels):
    skill_env.McClient = FakeClient
    env = AutoEnv(0, levels=levels)
    return env


def actions(rng, n):
    """Inputs that do things: long presses, clicks, turns, some of everything."""
    a = np.zeros((n, 9), np.int64)
    cur = np.array([1, 0, 0, 0, 0, 1, 5, 5, 0])
    for t in range(n):
        if rng.random() < 0.15:
            cur = np.array([rng.integers(3), rng.integers(3), rng.integers(2), 0, rng.integers(2), rng.integers(2), rng.integers(11), rng.integers(11), 0])
        a[t] = cur
        a[t, 3] = rng.random() < 0.3
        a[t, 8] = rng.random() < 0.4  # left clicks: hits when something is in reach
        if rng.random() < 0.5:
            a[t, 6] = a[t, 7] = 5
    return a


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(1)
    worst, ticks, bad, pushes = 0.0, 0, 0, 0
    families = ["hop", "path", "bridge", "bridge", "treenan", "stairs", "bedwars", "bedwars", "duel", "duel", "duel"]
    ref = None
    for n in range(rounds):
        levels = curriculum.Curriculum().levels
        family = families[n % len(families)]
        for d, values in curriculum.FAMILIES[family].items():
            if True:  # every dial, expert ones too
                levels[family][d] = int(rng.integers(len(values)))
        if ref is None:
            ref = reference_env(levels)
        ref.set_levels(levels)
        ref.set_exam(family)
        maker = CourseMaker(levels)
        maker.set_exam(family)
        seed = 1000 + n
        spec, meta = maker.draw(seed=seed)
        o_ref, _ = ref.reset(seed=seed)
        batch = mcsim.Batch(1)
        batch.set_course(0, spec)
        obs = np.zeros((1, mcsim.N_OBS), np.float32)
        batch.observe(0, obs[0])
        assert spec["goal"] == [ref.goal_x, ref.goal_z], (spec["goal"], ref.goal_x, ref.goal_z)
        diff = np.abs(obs[0] - o_ref).max()
        assert diff < 1e-5, f"round {n} {family}: first view differs by {diff} at {np.argmax(np.abs(obs[0] - o_ref))}"
        rew, done = np.zeros(1, np.float32), np.zeros(1, bool)
        for t, a in enumerate(actions(rng, 3000)):
            o_ref, r_ref, term, trunc, info = ref.step(a)
            batch.step(a[None], obs, rew, done)
            ticks += 1
            d = np.abs(obs[0] - o_ref)
            worst = max(worst, float(d.max()))
            if d.max() > 1e-4 or abs(rew[0] - r_ref) > 1e-5 or done[0] != (term or trunc):
                i = int(np.argmax(d))
                p = batch.player(0)
                print(f"round {n} {family} tick {t}: obs differs {d.max():.2e} at {i} (rust {obs[0][i]:.5f} python {o_ref[i]:.5f}), reward rust {rew[0]:.5f} python {r_ref:.5f}, "
                      f"done {done[0]} {term or trunc}, pos rust {p['pos']} python {ref.obs['pos']}")
                bad += 1
                break
            if done[0]:
                rep = batch.report(0)
                pushes += info["pushes"]
                for key in ("success", "clean", "fell", "deaths", "stalled", "blocks_used", "landed", "pushes"):
                    assert rep[key] == info[key], (n, key, rep[key], info[key])
                for k, v in info["parts"].items():
                    assert abs(rep["parts"].get(k, 0.0) - v) < 1e-6, (n, k, rep["parts"].get(k), v)
                break
    print(f"{rounds} rounds, {ticks} ticks, {pushes} pushes, largest difference {worst:.2e}, rounds that differ {bad}")
    assert bad == 0
    print("sim env ok")


if __name__ == "__main__":
    main()
