"""Pretrains the model by copying the scripted player, played in the real game. No human recording needed."""
import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from expert import act
from skill_env import JUMP, STAGES, SkillEnv

ROOT = Path(__file__).resolve().parent.parent
GAMMA = 0.99


def collect(env, rng, stages, episodes, noise):
    """Plays rounds with the script, sometimes doing a random action so it also shows how to recover.
    The label is always what the script would do in that state."""
    obs_all, act_all, ret_all = [], [], []
    for stage in stages:
        env.set_stage(stage)
        t0, clean = time.time(), 0
        for _ in range(episodes):
            obs, _ = env.reset()
            ep_obs, ep_act, ep_rew = [], [], []
            done, info = False, {}
            while not done:
                label = act(env)
                ep_obs.append(obs)
                ep_act.append(label)
                taken = env.action_space.sample() if rng.random() < noise else label
                obs, reward, term, trunc, info = env.step(taken)
                ep_rew.append(reward)
                done = term or trunc
            returns, g = [], 0.0
            for r in reversed(ep_rew):
                g = r + GAMMA * g
                returns.append(g)
            obs_all += ep_obs
            act_all += ep_act
            ret_all += returns[::-1]
            clean += bool(info["success"] and not info["deaths"])
        print(f"stage {stage + 1}: {clean}/{episodes} clean with noise, {len(obs_all)} examples so far, {time.time() - t0:.0f} s", flush=True)
    return np.array(obs_all, np.float32), np.array(act_all, np.int64), np.array(ret_all, np.float32)


class Spaces(gym.Env):
    observation_space = gym.spaces.Box(-5.0, 5.0, (SkillEnv.N_OBS,), np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 3, 2, 3, 2, 2, 7, 7])

    def reset(self, *, seed=None, options=None):
        return np.zeros(SkillEnv.N_OBS, np.float32), {}

    def step(self, action):
        return np.zeros(SkillEnv.N_OBS, np.float32), 0.0, False, False, {}


def train(obs, actions, returns, epochs, out):
    model = PPO("MlpPolicy", Spaces(), n_steps=2048, batch_size=512, n_epochs=8, gamma=GAMMA, learning_rate=3e-4, ent_coef=0.01,
                device="cpu", policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    n = len(obs)
    order = np.random.default_rng(0).permutation(n)
    val, tr = order[: n // 10], order[n // 10 :]
    o, a, r = torch.as_tensor(obs), torch.as_tensor(actions), torch.as_tensor(returns)
    names = ["move", "strafe", "sneak", "clicks", "jump", "sprint", "yaw", "pitch"]
    for epoch in range(epochs):
        policy.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 512):
            b = perm[i : i + 512]
            values, log_prob, _ = policy.evaluate_actions(o[b], a[b])
            loss = -log_prob.mean() + 0.5 * ((values.flatten() - r[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            policy.eval()
            with torch.no_grad():
                values, log_prob, _ = policy.evaluate_actions(o[val], a[val])
                pred = np.stack([c.probs.argmax(1).numpy() for c in policy.get_distribution(o[val]).distribution], 1)
            acc = (pred == actions[val]).mean(0)
            print(f"epoch {epoch}: copy loss {-log_prob.mean().item():.3f}, value error {(values.flatten() - r[val]).abs().mean().item():.3f}, "
                  f"match {dict(zip(names, acc.round(2).tolist()))}", flush=True)
    model.save(out)
    return model


def evaluate(env, model, stages, episodes):
    for stage in stages:
        env.set_stage(stage)
        clean = died = stuck = 0
        speed, shift = [], []
        for _ in range(episodes):
            obs, _ = env.reset()
            done, info = False, {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            ok = info["success"] and not info["deaths"]
            clean += ok
            died += info["deaths"] > 0
            stuck += info["stalled"]
            shift.append(info["shift_share"])
            if ok:
                speed.append(info["blocks_per_s"])
        t = STAGES[stage]
        print(f"model stage {stage + 1:2d} gap {t[0]} {'jump' if t[6] == JUMP else 'build'} ±{t[3]:3d} {t[5]:6s}: {clean}/{episodes} clean, {died} died, {stuck} stuck, "
              f"{np.mean(speed) if speed else 0:.2f} blocks/s, shift {np.mean(shift):.0%}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=25601)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--out", default=str(ROOT / "runs" / "models" / "pretrained"))
    a = p.parse_args()
    env = SkillEnv(a.port)
    rng = np.random.default_rng(0)
    stages = [i for i, t in enumerate(STAGES) if t[1] == "axis"]
    data = ROOT / "runs" / "models" / "pretrain_examples.npz"
    obs, actions, returns = collect(env, rng, stages, a.episodes, a.noise)
    np.savez_compressed(data, obs=obs, actions=actions, returns=returns)
    print(f"{len(obs)} examples saved", flush=True)
    torch.set_num_threads(4)
    model = train(obs, actions, returns, a.epochs, a.out)
    evaluate(env, model, stages, 20)
    env.c.close()
