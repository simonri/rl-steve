"""What does the model do on a jump stage? Measures inputs and positions, no training."""
import argparse
from collections import Counter

import math

import numpy as np
import torch
from stable_baselines3 import PPO

from skill_env import STAGES, SkillEnv

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--port", type=int, default=25601)
p.add_argument("--stage", type=int, default=5)
p.add_argument("--episodes", type=int, default=40)
a = p.parse_args()
env = SkillEnv(a.port, stage=a.stage)
model = PPO.load(a.model, device="cpu")
NAMES = ["move", "strafe", "sneak", "use", "jump", "sprint", "yaw", "pitch"]
print("stage", STAGES[a.stage])
for det in (False, True):
    ends, picks, jumps, jump_edge, reached_edge, air, sprint_moving = Counter(), [Counter() for _ in NAMES], [], [], 0, 0, 0
    closest, ticks = [], []
    aim = []
    probs = None
    for _ in range(a.episodes):
        obs, _ = env.reset()
        if probs is None:
            with torch.no_grad():
                d = model.policy.get_distribution(torch.as_tensor(obs[None]))
            probs = [c.probs[0].numpy().round(2).tolist() for c in d.distribution]
        done, n_jump, best_edge = False, 0, 99.0
        axis = 0 if env.goal_x else 2
        sign = 1 if (env.goal_x + env.goal_z) > 0 else -1
        edge = 2.0 if sign > 0 else -1.0
        while not done:
            act, _ = model.predict(obs, deterministic=det)
            for i, v in enumerate(act):
                picks[i][int(v)] += 1
            before = env.obs["on_ground"]
            obs, _, term, trunc, info = env.step(act)
            left = (edge - env.obs["pos"][axis]) * sign
            best_edge = min(best_edge, left)
            if act[4] and before and not env.obs["on_ground"] and env.obs["vel"][1] > 0.2:
                n_jump += 1
                jump_edge.append(round(left, 2))
                x_, z_ = env.obs["pos"][0], env.obs["pos"][2]
                want = math.degrees(math.atan2(-(env.goal_x + 0.5 - x_), env.goal_z + 0.5 - z_))
                aim.append(abs((env.obs["yaw"] - want + 180) % 360 - 180))
            sprint_moving += int(act[5] and act[0] == 1)
            done = term or trunc
        ends["goal" if info["success"] else "fell" if info["fell"] else "stuck" if info["stalled"] else "timeout"] += 1
        jumps.append(n_jump)
        closest.append(best_edge)
        ticks.append(env.t)
    print(f"--- {'best action' if det else 'sampled'}: {dict(ends)} mean ticks {np.mean(ticks):.0f}")
    print("first-state probabilities:", dict(zip(NAMES, probs)))
    for name, c in zip(NAMES, picks):
        total = sum(c.values())
        print(f"  {name:7s}", {k: round(v / total, 2) for k, v in sorted(c.items())})
    print(f"  jumps per episode {np.mean(jumps):.2f}, distance to edge when jumping {sorted(jump_edge)[:12]}")
    if aim:
        print(f"  aim error at jump: median {np.median(aim):.0f} deg, within 15 deg {np.mean(np.array(aim) <= 15):.0%}, over 45 deg {np.mean(np.array(aim) > 45):.0%}")
    print(f"  closest to edge per episode: mean {np.mean(closest):.2f}, min {np.min(closest):.2f}; decisions with sprint+forward {sprint_moving}")
env.c.close()
