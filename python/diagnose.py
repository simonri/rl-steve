"""Scores a saved model per direction and height, to find where it loses points. No training."""
import argparse
import math

from stable_baselines3 import PPO

from skill_env import FLAT, HILLS, SkillEnv

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--port", type=int, default=25601)
p.add_argument("--stage", type=int, default=9)
p.add_argument("--episodes", type=int, default=40)
a = p.parse_args()


class Fixed(SkillEnv):
    angle, heights_override = 0.0, FLAT

    def _direction(self):
        return math.cos(self.angle), math.sin(self.angle)

    def _use_task(self, stage):
        super()._use_task(stage)
        self.heights = self.heights_override


env = Fixed(a.port, stage=a.stage)
model = PPO.load(a.model, device="cpu")
print("direction height | goal fell stuck | speed | used needed ratio unused manhattan-ratio")
for name, angle in (("straight", 0.0), ("22deg", math.pi / 8), ("diagonal", math.pi / 4)):
    for hname, heights in (("flat", FLAT), ("up/down", HILLS)):
        env.angle, env.heights_override = angle, heights
        env.set_stage(a.stage)
        wins, fell, stuck, used, needed, unused, speed, man = 0, 0, 0, 0, 0, 0, 0.0, 0.0
        for _ in range(a.episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(act)
                done = term or trunc
            fell += info["fell"]
            stuck += info["stalled"]
            if info["success"]:
                wins += 1
                used += info["blocks_used"]
                needed += info["blocks_needed"]
                unused += info["unused"]
                speed += info["blocks_per_s"]
                man += max(abs(env.goal_x) - 4, 0) + max(abs(env.goal_z) - 4, 0)
        w = max(wins, 1)
        print(f"{name:9s} {hname:8s}| {wins}/{a.episodes} {fell} {stuck} | {speed / w:.2f} | {used / w:.1f} {needed / w:.1f} {used / max(needed, 1):.2f} {unused / w:.1f} {used / max(man, 1):.2f}", flush=True)
