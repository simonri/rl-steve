"""Does the recorded play help? Copies the model, fine-tunes the copy on the recordings, and plays both on the same courses.

The copy also has to keep answering like the original on the model's own play, so that it does not forget the rest.
Nothing here touches the running training.
"""
import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import demo_data
from auto_env import AutoEnv
from curriculum import FAMILIES, is_open

ROOT = Path(__file__).resolve().parent.parent
HEADS = ("move", "strafe", "sneak", "clicks", "jump", "sprint", "yaw", "pitch")
ROUNDS = {"hop": 100, "path": 100, "bridge": 60, "treenan": 40}


def play(env, model, levels, seen=None):
    """Same courses for every model: each round's random numbers come from its number."""
    out = {}
    for family in FAMILIES:
        if not is_open(levels, family):
            continue
        env.set_levels(levels)
        env.set_exam(family)
        clean = landed = 0
        for number in range(ROUNDS[family]):
            env._np_random = np.random.default_rng(10_000 + number)
            obs, _ = env.reset()
            done, info = False, {}
            while not done:
                if seen is not None:
                    seen.append(obs.copy())
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            clean += bool(info["clean"])
            landed += info["landed"]
        out[family] = (clean, ROUNDS[family])
        print(f"  {family}: {clean}/{ROUNDS[family]} clean, {landed / ROUNDS[family]:.2f} blocks per round", flush=True)
    return out


def agreement(policy, obs, actions):
    with torch.no_grad():
        chosen = torch.stack([d.probs.argmax(1) for d in policy.get_distribution(obs).distribution], 1)
    return (chosen == actions).float().mean(0).numpy()


def fine_tune(model, demos, anchor, steps, weight):
    policy = model.policy
    files = sorted(demos)
    np.random.default_rng(0).shuffle(files)
    held = max(1, len(files) // 5)
    pairs = [demo_data.replay(f)[:2] for f in files]
    val_o = torch.as_tensor(np.concatenate([p[0] for p in pairs[:held]]))
    val_a = torch.as_tensor(np.concatenate([p[1] for p in pairs[:held]]))
    o = torch.as_tensor(np.concatenate([p[0] for p in pairs[held:]]))
    a = torch.as_tensor(np.concatenate([p[1] for p in pairs[held:]]))
    print(f"recordings: {len(o)} ticks to learn from, {len(val_o)} held back", flush=True)
    print("matches the player, before:", dict(zip(HEADS, agreement(policy, val_o, val_a).round(2).tolist())), flush=True)
    anchor = torch.as_tensor(np.asarray(anchor, np.float32))
    with torch.no_grad():
        target = [d.logits.detach() for d in policy.get_distribution(anchor).distribution]
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    rng = np.random.default_rng(1)
    for step in range(steps):
        b = rng.integers(0, len(o), 256)
        k = rng.integers(0, len(anchor), 256)
        _, log_prob, _ = policy.evaluate_actions(o[b], a[b])
        copied = -log_prob.mean()
        now = policy.get_distribution(anchor[k]).distribution
        keep = sum(torch.distributions.kl_divergence(torch.distributions.Categorical(logits=t[k]), d).mean() for t, d in zip(target, now))
        loss = copied + weight * keep
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  step {step}: copying {copied.item():.3f}, drift from the original {keep.item():.3f}", flush=True)
    print("matches the player, after: ", dict(zip(HEADS, agreement(policy, val_o, val_a).round(2).tolist())), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=25605)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--weight", type=float, default=1.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    models = ROOT / "runs" / "models" / "auto1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start = out / "start.zip"
    start.write_bytes((models / "latest.zip").read_bytes())
    levels = json.loads((models / "levels.json").read_text())
    (out / "levels.json").write_text(json.dumps(levels))
    env = AutoEnv(a.port)
    original = PPO.load(start, device="cpu")
    seen = []
    print("original model:", flush=True)
    before = play(env, original, levels, seen)
    tuned = PPO.load(start, device="cpu")
    fine_tune(tuned, glob(str(ROOT / "runs" / "demos" / "*" / "round*.jsonl.gz")), seen, a.steps, a.weight)
    tuned.save(out / "tuned")
    print("fine-tuned on the recordings:", flush=True)
    after = play(env, tuned, levels)
    print("result:", {f: (before[f], after[f]) for f in before}, flush=True)
    env.c.close()
