"""Real game check: a scripted runner on the winding paths, with all human limits. No AI."""
import _path  # noqa: F401  (python/ on the import path)
import sys

import numpy as np

from skill_env import RUN, RUN_HOLES, STAGES, YAW_STEPS, SkillEnv

env = SkillEnv(int(sys.argv[1]))
EPISODES = int(sys.argv[2])


def turn_index(error):
    return min(range(len(YAW_STEPS)), key=lambda i: abs(error / 2 - YAW_STEPS[i]))


def runner(sprint):
    ahead, side = env._way_in_view(env.obs)
    error = np.degrees(np.arctan2(-side, ahead)) * -1
    near_ahead, near_side = env._look_ahead(env.obs)
    bend = abs(near_side) > 0.25
    edge = env._edge_left(env.obs) * 2
    jump = int(edge < 0.5 and env.extras == RUN_HOLES and not bend)
    return [1 if ahead > 0.6 else 0, 0, 0, 0, jump, int(sprint and not bend), turn_index(error), 3]


for kind in (RUN, RUN_HOLES):
    stage = next(i for i, t in enumerate(STAGES) if t[6] == kind)
    env.set_stage(stage)
    env.delay_low, env.delay_high, env.late_chance, env.aim_noise = 3, 5, 0.3, 0.3
    for sprint in (True, False):
        clean = died = stuck = 0
        speed, holes, lengths = [], [], []
        for _ in range(EPISODES):
            env.reset()
            lengths.append(env.par)
            done, info = False, {}
            while not done:
                _, _, term, trunc, info = env.step(runner(sprint))
                done = term or trunc
            ok = info["success"] and not info["deaths"]
            clean += ok
            died += info["deaths"] > 0
            stuck += info["stalled"]
            if ok:
                speed.append(info["blocks_per_s"])
        print(f"{kind:10s} sprint={sprint!s:5s}: {clean}/{EPISODES} clean, {died} died, {stuck} stuck, {np.mean(speed) if speed else 0:.2f} blocks/s, path cells ~{np.mean(lengths):.0f}", flush=True)
env.c.close()
print("ok")
