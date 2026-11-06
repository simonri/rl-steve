"""How well does the hand-written player do in the real game, with the stage's limits? No AI."""
import _path  # noqa: F401  (python/ on the import path)
import sys

import numpy as np

from expert import act
from skill_env import JUMP, STAGES, SkillEnv

env = SkillEnv(int(sys.argv[1]))
stages = [i for i, t in enumerate(STAGES) if t[1] == "axis"]
rows = []
for stage in stages:
    env.set_stage(stage)
    wins = deaths = stuck = 0
    speeds, shift = [], []
    for _ in range(int(sys.argv[2])):
        env.reset()
        done, info = False, {}
        while not done:
            _, _, term, trunc, info = env.step(act(env))
            done = term or trunc
        clean = info["success"] and not info["deaths"]
        wins += clean
        deaths += info["deaths"] > 0
        stuck += info["stalled"]
        shift.append(info["shift_share"])
        if clean:
            speeds.append(info["blocks_per_s"])
    t = STAGES[stage]
    print(f"stage {stage + 1:2d} gap {t[0]} {'jump' if t[6] == JUMP else 'build'} angle ±{t[3]:3d} {t[5]:6s}: {wins}/{sys.argv[2]} clean, {deaths} died, {stuck} stuck, "
          f"{np.mean(speeds) if speeds else 0:.2f} blocks/s, shift {np.mean(shift):.0%}", flush=True)
env.c.close()
