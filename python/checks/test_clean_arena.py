"""Real game check: after every reset, only the new arena may exist in the world."""
import _path  # noqa: F401  (python/ on the import path)
import sys

import numpy as np

from skill_env import Y, SkillEnv

R = 45
env = SkillEnv(int(sys.argv[1]))


def solid_outside_arena(spec_boxes):
    bad = []
    for x in range(-R, R + 1, 1):
        names = env.c.call(op="blocks", min=[x, Y - 16, -R], max=[x, Y + 16, R])["names"]
        i = 0
        for y in range(Y - 16, Y + 17):
            for z in range(-R, R + 1):
                if names[i] not in ("minecraft:air", "minecraft:void_air", "minecraft:cave_air"):
                    if not any(b[0] <= x <= b[3] and b[1] <= y <= b[4] and b[2] <= z <= b[5] for b in spec_boxes):
                        bad.append((x, y, z))
                i += 1
    return bad


real_spec = env._spec
last = {}


def remember():
    s = real_spec()
    last.update(s)
    return s


env._spec = remember
rng = np.random.default_rng(0)
for stage in (6, 10, 13, 16, 20):
    env.set_stage(stage)
    for episode in range(4):
        env.reset()
        bad = solid_outside_arena(last["platforms"])
        assert not bad, (stage, episode, bad[:5], len(bad))
        # Build what can be built, so the next reset has something to clean up.
        for _ in range(80):
            act = env.action_space.sample()
            act[2], act[3] = 1, 2
            _, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
    print(f"stage {stage + 1} clean after {episode + 1} resets", flush=True)
print("ok")
env.c.close()
