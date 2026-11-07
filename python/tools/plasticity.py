"""Has the network lost its ability to learn new things (plasticity)? Signs of it, per model, on the same inputs:

- dormant units: the share of a layer's units whose mean |activation| is under 10 % of the layer's average (Sokar et
  al. 2023, "The Dormant Neuron Phenomenon in Deep RL"): units that no longer respond cannot learn either;
- effective rank of the features the policy and value read: how many directions the network still uses (the number of
  singular values that make up 99 % of their sum, Kumar et al. 2021);
- the weights' size: weights that keep growing make every update relatively smaller.

A freshly initialized network is the reference. Juliani and Ash (2024, arXiv 2405.19153) found plasticity loss in
on-policy RL, and that most fixes from elsewhere fail there; this measures whether it is our problem at all.

    python tools/plasticity.py MODEL.zip [MODEL.zip ...]
"""
import _path  # noqa: F401
import json
import sys

import numpy as np
import torch

import curriculum
import train_auto as ta
from sim_env import SimVecEnv
from spatial import MemoryPolicy

TAU = 0.1
ROUNDS, TICKS = 128, 120


def inputs(levels):
    """Inputs from rounds of every kind, played by a fresh network (the same inputs for every model measured)."""
    vec = SimVecEnv(ROUNDS, levels, seed=31337)
    rng = np.random.default_rng(0)
    obs = vec.reset()
    seen = []
    for _ in range(TICKS):
        seen.append(obs.copy())
        a = np.stack([rng.integers(c, size=ROUNDS) for c in [3, 3, 2, 2, 2, 2, 11, 11, 2, 10]], 1)
        a[:, 9] = 0
        a[:, 6] = np.where(rng.random(ROUNDS) < 0.7, 5, a[:, 6])
        a[:, 7] = np.where(rng.random(ROUNDS) < 0.7, 5, a[:, 7])
        vec.step_async(a)
        obs, _, _, _ = vec.step_wait()
    return torch.as_tensor(np.concatenate(seen))


def measure(policy, obs, device):
    acts = {}
    hooks = []
    for name, mod in policy.named_modules():
        if isinstance(mod, (torch.nn.SiLU, torch.nn.ReLU, torch.nn.Tanh)) or name.endswith(("cube_out", "map_out")):
            hooks.append(mod.register_forward_hook(lambda m, i, o, name=name: acts.setdefault(name, []).append(o.detach().abs().flatten(2).mean(2) if o.dim() > 2 else o.detach().abs())))
    lstm = policy.lstm_actor
    n = 512
    feats = []
    with torch.no_grad():
        for i in range(0, len(obs), n):
            x = obs[i : i + n].to(device)
            z = (torch.zeros(lstm.num_layers, len(x), lstm.hidden_size, device=device),) * 2
            f = policy.extract_features(x, policy.pi_features_extractor if hasattr(policy, "pi_features_extractor") else None)
            feats.append(f.cpu())
            policy.get_distribution(x, z, torch.ones(len(x), device=device))
    for h in hooks:
        h.remove()
    out = {}
    dormant_all = []
    for name, chunks in acts.items():
        a = torch.cat(chunks).mean(0)  # mean |activation| per unit (channels for convolutions)
        score = a / a.mean().clamp_min(1e-12)
        d = float((score <= TAU).float().mean())
        out[name or "out"] = round(d, 3)
        dormant_all.append((d, a.numel()))
    f = torch.cat(feats)
    s = torch.linalg.svdvals(f - f.mean(0))
    srank = int((torch.cumsum(s, 0) / s.sum() < 0.99).sum()) + 1
    wnorm = float(sum((p.detach() ** 2).sum() for p in policy.parameters()) ** 0.5)
    total = sum(d * k for d, k in dormant_all) / max(sum(k for _, k in dormant_all), 1)
    return dict(dormant=round(total, 3), feature_rank=srank, weight_norm=round(wnorm, 1), layers=out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    levels = json.load(open("/home/simon/dev/minecraft-ai-train/runs/models/auto7/levels.json"))
    obs = inputs(levels)
    vec = SimVecEnv(1, levels, seed=0)
    fresh = ta.SplitPPO(MemoryPolicy, vec, n_steps=8, batch_size=8, policy_kwargs=ta.POLICY, device=device, seed=0).policy
    fresh.set_training_mode(False)
    rows = [("fresh network", measure(fresh, obs, device))]
    for path in sys.argv[1:]:
        rows.append((path, measure(ta.load_policy(path, device), obs, device)))
    for name, r in rows:
        print(f"{name}\n  dormant {r['dormant']:.3f}  feature rank {r['feature_rank']}/256  weight norm {r['weight_norm']}")
        print("  by layer: " + ", ".join(f"{k} {v}" for k, v in r["layers"].items()))


if __name__ == "__main__":
    main()
