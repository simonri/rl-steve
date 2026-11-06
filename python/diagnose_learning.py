"""How set in its ways is the model? Entropy, action choices, value estimates and network health. No training."""
import argparse
import math

import numpy as np
import torch
from stable_baselines3 import PPO

from skill_env import SkillEnv

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--port", type=int, default=25601)
p.add_argument("--stage", type=int, default=9)
p.add_argument("--episodes", type=int, default=40)
p.add_argument("--best", action="store_true")
a = p.parse_args()
env = SkillEnv(a.port, stage=a.stage)
model = PPO.load(a.model, device="cpu")
print("stage", a.stage + 1, "lr", model.learning_rate, "ent", model.ent_coef, "gamma", model.gamma, "timesteps", model.num_timesteps)
NAMES = ["move", "strafe", "sneak", "clicks", "jump", "sprint", "yaw", "pitch"]
MAX = [math.log(n) for n in (3, 3, 2, 3, 2, 2, 7, 5)]
ents, jumpp, sneakp, clickp, values, returns, ends = [], [], [], [], [], [], {"goal": 0, "fell": 0, "stuck": 0, "timeout": 0}
placed, jumped, starts = [], [], []
facing, back_key, forward_key, deaths, first_death = [], 0, 0, [], []
for _ in range(a.episodes):
    obs, _ = env.reset()
    done, reward_sum, first_value, n_jump = False, 0.0, None, 0
    while not done:
        t = torch.as_tensor(obs[None])
        with torch.no_grad():
            d = model.policy.get_distribution(t)
            v = model.policy.predict_values(t).item()
        if first_value is None:
            first_value = v
        cats = d.distribution
        ents.append([c.entropy().item() for c in cats])
        jumpp.append(cats[4].probs[0, 1].item())
        sneakp.append(cats[2].probs[0, 1].item())
        clickp.append(1 - cats[3].probs[0, 0].item())
        act, _ = model.predict(obs, deterministic=a.best)
        facing.append(env._way_in_view(env.obs)[0])
        back_key += int(act[0] == 2)
        forward_key += int(act[0] == 1)
        before = env.obs["on_ground"]
        obs, r, term, trunc, info = env.step(act)
        reward_sum += r
        if act[4] and before and not env.obs["on_ground"] and env.obs["vel"][1] > 0.2:
            n_jump += 1
        done = term or trunc
    values.append(first_value)
    returns.append(reward_sum)
    placed.append(info["blocks_used"])
    deaths.append(info["deaths"])
    jumped.append(n_jump)
    ends["goal" if info["success"] else "fell" if info["fell"] else "stuck" if info["stalled"] else "timeout"] += 1
print("episodes", ends, f"blocks placed per episode {np.mean(placed):.2f}, jumps per episode {np.mean(jumped):.2f}")
e = np.array(ents).mean(0)
f_ = np.array(facing)
print(f"facing the goal (over 0.7): {np.mean(f_ > 0.7):.0%}, back to the goal (under -0.7): {np.mean(f_ < -0.7):.0%}; move keys forward {forward_key} back {back_key}; deaths per episode {np.mean(deaths):.2f}")
print("entropy as share of maximum:", {n: round(float(x / m), 2) for n, x, m in zip(NAMES, e, MAX)})
print(f"mean P(jump) {np.mean(jumpp):.2f}, P(shift) {np.mean(sneakp):.2f}, P(click) {np.mean(clickp):.2f}")
print(f"value at episode start {np.mean(values):.2f} vs actual return {np.mean(returns):.2f}")
w = [p_.detach().numpy() for n, p_ in model.policy.named_parameters() if "weight" in n]
print("largest weight per layer:", [round(float(np.abs(x).max()), 2) for x in w], "mean abs:", [round(float(np.abs(x).mean()), 3) for x in w])
env.c.close()
