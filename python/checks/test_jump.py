"""Real game check: can the zero-block jump stages be cleared with all human limits on? Scripted, no AI."""
import _path  # noqa: F401  (python/ on the import path)
import math
import sys

from skill_env import JUMP, STAGES, SkillEnv

env = SkillEnv(int(sys.argv[1]))
stages = [i for i, t in enumerate(STAGES) if t[6] == JUMP]
assert len(stages) == 5, stages
for stage in (stages[0], stages[-1]):
    env.set_stage(stage)
    env.delay_low, env.delay_high, env.late_chance, env.aim_noise = 3, 5, 0.3, 0.3
    for sprint in (1, 0):
        for margin in (0.3, 0.6, 1.0):
            wins, falls = 0, 0
            for _ in range(12):
                spec = env._spec()
                assert spec["blocks"] == 0
                ux, uz = math.copysign(1, env.goal_x) if env.goal_x else 0, math.copysign(1, env.goal_z) if env.goal_z else 0
                spec["yaw"] = math.degrees(math.atan2(-ux, uz)) % 360
                spec["pitch"] = 0.0
                env.par = 1
                env.obs = env.c.reset(**spec)
                env.limit = 300
                env._begin_episode()
                done, info = False, {}
                while not done:
                    x, _, z = env.obs["pos"]
                    edge = 2.0 if ux + uz > 0 else -1.0
                    along = x if ux else z
                    left = (edge - along) * (ux + uz)
                    jump = int(left < margin)
                    _, _, term, trunc, info = env.step([1, 0, 0, 0, jump, sprint, 3, 3])
                    done = term or trunc
                wins += info["success"]
                falls += info["fell"]
            print(f"gap {env.gap} sprint={sprint} jump {margin} before edge: {wins}/12 goal, {falls} fell", flush=True)
env.c.close()
print("ok")
