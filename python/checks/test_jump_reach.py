"""Real game check: which gaps can be cleared by a sprint jump alone? Decides where jumping stops working."""
import _path  # noqa: F401  (python/ on the import path)
import math
import sys

from skill_env import FLAT, SkillEnv, Y


class Fixed(SkillEnv):
    angle = 0.0

    def _direction(self):
        return math.cos(self.angle), math.sin(self.angle)


env = Fixed(int(sys.argv[1]))
env.set_stage(0)
env.delay_low, env.delay_high, env.late_chance, env.aim_noise = 3, 5, 0.3, 0.3
for gap in (2, 3, 4, 5):
    for name, angle in (("straight", 0.0), ("diagonal", math.pi / 4)):
        best = 0
        for margin in (0.2, 0.4, 0.6, 0.9):
            wins = 0
            for _ in range(10):
                env.angle = angle
                env.gap, env.heights, env.directions, env.yaw_range, env.start, env.extras, env.budget = gap, FLAT, "any", 0, "center", None, (0.0, 0)
                spec = env._spec()
                ux, uz = math.cos(angle), math.sin(angle)
                spec["yaw"] = math.degrees(math.atan2(-ux, uz)) % 360
                spec["pitch"] = 0.0
                # Longest run-up the platform allows, from the far corner.
                spec["pos"] = [0.5 - ux * 1.1, Y + 1, 0.5 - uz * 1.1]
                env.obs = env.c.reset(**spec)
                env.limit = 300
                env._begin_episode()
                done, info = False, {}
                while not done:
                    x, _, z = env.obs["pos"]
                    if angle == 0.0:
                        left = 2.0 - x
                    else:
                        left = min(2.0 - x, 2.0 - z)
                    _, _, term, trunc, info = env.step([1, 0, 0, 0, int(left < margin), 1, 3, 3])
                    done = term or trunc
                wins += info["success"]
            best = max(best, wins)
        print(f"gap {gap} {name}: goal x={env.goal_x} z={env.goal_z}, best {best}/10", flush=True)
env.c.close()
