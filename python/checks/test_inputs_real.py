"""Real game check: every input is finite and within range on all stages, and the new ones make sense."""
import _path  # noqa: F401  (python/ on the import path)
import sys

import numpy as np

from skill_env import HEAD, KNOWN_SLOT, STAGES, SkillEnv

env = SkillEnv(int(sys.argv[1]))
ranges = np.zeros((HEAD, 2))
ranges[:, 0], ranges[:, 1] = 1e9, -1e9
edge_seen, gain_pos, gain_neg = set(), 0, 0
for stage in range(len(STAGES)):
    env.set_stage(stage)
    for _ in range(3):
        obs, _ = env.reset()
        assert obs.shape == (SkillEnv.N_OBS,) and np.isfinite(obs).all(), stage
        for _ in range(25):
            act = env.action_space.sample()
            obs, r, term, trunc, info = env.step(act)
            assert np.isfinite(obs).all() and np.isfinite(r), (stage, obs)
            ranges[:, 0] = np.minimum(ranges[:, 0], obs[:HEAD])
            ranges[:, 1] = np.maximum(ranges[:, 1], obs[:HEAD])
            edge_seen.add(round(float(obs[KNOWN_SLOT + 5]), 1))
            gain_pos += obs[KNOWN_SLOT + 6] > 0.5
            gain_neg += obs[KNOWN_SLOT + 6] < -0.5
            if term or trunc:
                break
print("largest input size:", round(float(np.abs(ranges).max()), 2))
print("values by slot:", [f"{a:.1f}..{b:.1f}" for a, b in ranges])
print(f"edge values seen {sorted(edge_seen)[:12]}, place gain + {gain_pos} - {gain_neg}")
assert np.abs(ranges).max() < 6
assert len(edge_seen) > 3 and gain_pos and gain_neg
env.c.close()
print("ok")
