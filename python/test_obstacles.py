"""Real game check: every obstacle stage builds, resets cleanly and can be stepped through."""
import sys

import numpy as np

from skill_env import NO_BLOCKS, STAGES, SkillEnv

env = SkillEnv(int(sys.argv[1]))
rng = np.random.default_rng(0)
for stage, task in enumerate(STAGES):
    if not task[6] or task[6] in NO_BLOCKS or task[6] == "treenan":
        continue
    env.set_stage(stage)
    kinds = {}
    for _ in range(12):
        obs, _ = env.reset()
        assert obs.shape == (SkillEnv.N_OBS,) and np.isfinite(obs).all()
        kinds[env.kind] = kinds.get(env.kind, 0) + 1
        for name, blocks in (("stone", env.c.call(op="blocks", min=[env.goal_x, env.dy + 100, env.goal_z], max=[env.goal_x, env.dy + 100, env.goal_z])["names"]),):
            assert blocks == ["minecraft:stone"], blocks
        for x, z in list(env.blocked)[:1]:
            names = env.c.call(op="blocks", min=[x, 100, z], max=[x, 100, z])["names"]
            assert names == ["minecraft:stone"], (x, z, names)
        start = env._distance(env.obs)
        for _ in range(30):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(obs).all() and np.isfinite(r)
            if term or trunc:
                break
    print(f"stage {stage + 1}: {task[6]} ok, kinds {kinds}, needed blocks {env.par}", flush=True)
env.c.close()
print("ok")
