"""A fixed test that never changes, to see real progress over a run (and between runs).

Why: the exams play every dial at its current limit, so they get harder each time the curriculum moves on, and a
score that falls can mean a better model on harder rounds. With 32 rounds their noise was +-0.08, as large as most
changes (auto7: path's exam swung 0.50-0.94 at one level). This plays the same rounds every time (fixed seeds),
at three fixed difficulties per course kind, with enough rounds for a 95 % interval of about +-0.06, and reports the
interval with every number (docs/training-design.md#mata).

- tiers: every dial at a third, two thirds, and all of its range (the dials that open last stay at 0 below the top)
- each of the n rounds counts once, its first (counting the first n to finish counts quick rounds twice)
- played sampled, as in training, and by best action, as the exams and the viewer play
- skills the points do not show: hits a second, critical and long hits, missed swings, sword hits, and how often and
  how soon it finds its blocks when a course starts with something else in hand

    python benchmark.py ../runs/models/auto7/latest.zip [--rounds 128] [--out ../runs/models/auto7/benchmark.jsonl]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import curriculum
from sim_env import SimVecEnv

FAMILIES = ("hop", "path", "bridge", "treenan", "stairs", "bedwars", "duel")
TIERS = {"easy": 1 / 3, "mid": 2 / 3, "hard": 1.0}
ROUNDS = 128
SEED = 7_000_000  # the rounds of every benchmark: never change it, or old results stop comparing


def wilson(k, n, z=1.96):
    """A 95 % interval for a rate seen k times in n (Wilson): honest also near 0 and 1."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tier_levels(family, share):
    """Every dial of `family` at `share` of its range; the dials that open last stay at 0 below the top tier."""
    levels = curriculum.Curriculum().levels
    late = curriculum.EXPERT.get(family, ())
    for dial, values in curriculum.FAMILIES[family].items():
        top = len(values) - 1
        levels[family][dial] = top if share >= 1.0 else 0 if dial in late else int(round(share * top))
    return levels


def first_rounds(vec, policy, device, deterministic, opponents=None):
    """Plays every round of `vec` once from a reset and returns each round's info (the first one to end in each)."""
    vec.opponents = opponents
    obs = vec.reset()
    n = vec.num_envs
    lstm = policy.lstm_actor
    memory = (torch.zeros(lstm.num_layers, n, lstm.hidden_size, device=device), torch.zeros(lstm.num_layers, n, lstm.hidden_size, device=device))
    starts = torch.ones(n, device=device)
    out = [None] * n
    with torch.no_grad():
        while any(o is None for o in out):
            dist, memory = policy.get_distribution(torch.as_tensor(obs, device=device), memory, starts)
            vec.step_async(dist.get_actions(deterministic=deterministic).cpu().numpy())
            obs, _, dones, infos = vec.step_wait()
            starts = torch.as_tensor(dones, dtype=torch.float32, device=device)
            for i in np.flatnonzero(dones):
                if out[i] is None:
                    out[i] = infos[i]
    vec.opponents = None
    return out


def summarize(infos):
    """Success with its interval, and the skills summed over the rounds."""
    n = len(infos)
    k = sum(bool(i["clean"]) for i in infos)
    lo, hi = wilson(k, n)
    s = {key: sum(i["skills"][key] for i in infos) for key in ("swings", "misses", "hits", "crits", "long_hits", "sword_hits")}
    ticks = sum(i["ticks"] for i in infos)
    out = dict(rounds=n, clean=round(k / n, 3), low=round(lo, 3), high=round(hi, 3))
    if s["swings"]:
        out.update(hits_per_s=round(20 * s["hits"] / max(ticks, 1), 3), miss_share=round(s["misses"] / s["swings"], 3),
                   crit_share=round(s["crits"] / max(s["hits"], 1), 3), long_share=round(s["long_hits"] / max(s["hits"], 1), 3),
                   sword_share=round(s["sword_hits"] / max(s["hits"], 1), 3))
    needed = [i for i in infos if i["skills"]["needed_blocks"]]
    if needed:
        took = [i["skills"]["took_blocks_at"] for i in needed if i["skills"]["took_blocks_at"] >= 0]
        lo, hi = wilson(len(took), len(needed))
        out.update(took_blocks=round(len(took) / len(needed), 3), took_low=round(lo, 3), took_high=round(hi, 3),
                   took_median_ticks=float(np.median(took)) if took else None)
    return out


def run(policy, device, rounds=ROUNDS, modes=("sampled", "best")):
    """The whole benchmark: {family: {tier: {mode: summary}}} plus the hand rounds."""
    result = {}
    for f_i, family in enumerate(FAMILIES):
        result[family] = {}
        for t_i, (tier, share) in enumerate(TIERS.items()):
            vec = SimVecEnv(rounds, tier_levels(family, share), seed=SEED + 100 * f_i + t_i)
            vec.env_method("set_exam", family, True)
            result[family][tier] = {m: summarize(first_rounds(vec, policy, device, m == "best")) for m in modes}
    # Finding the blocks: bridge at two thirds, every course starting with something else in hand.
    vec = SimVecEnv(rounds, tier_levels("bridge", 2 / 3), seed=SEED + 999)
    vec.env_method("set_exam", "bridge", True)
    vec.set_attr("hand", "random")
    result["hand"] = {"bridge mid": {m: summarize(first_rounds(vec, policy, device, m == "best")) for m in modes}}
    return result


def show(result):
    for family, tiers in result.items():
        for tier, modes in tiers.items():
            line = "  ".join(f"{m} {s['clean']:.2f} [{s['low']:.2f}-{s['high']:.2f}]" for m, s in modes.items())
            extra = modes.get("sampled", {})
            more = ""
            if "hits_per_s" in extra:
                more = f"  hits/s {extra['hits_per_s']} miss {extra['miss_share']} crit {extra['crit_share']} long {extra['long_share']} sword {extra['sword_share']}"
            if "took_blocks" in extra:
                more += f"  took blocks {extra['took_blocks']} [{extra['took_low']}-{extra['took_high']}] median {extra['took_median_ticks']} ticks"
            print(f"{family:8s} {tier:10s} {line}{more}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--out", help="a .jsonl to add this result to")
    p.add_argument("--modes", default="sampled,best", help="sampled (as in training), best (as exams and the viewer), or both")
    a = p.parse_args()
    import train_auto as ta

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ta.load_policy(a.model, device)
    started = time.time()
    result = run(policy, device, a.rounds, tuple(a.modes.split(",")))
    show(result)
    print(f"{time.time() - started:.0f} s")
    if a.out:
        steps = ta.SplitPPO.load(a.model, device="cpu", custom_objects=ta.QUIET_LOAD).num_timesteps
        with open(a.out, "a") as f:
            f.write(json.dumps(dict(model=str(Path(a.model).resolve()), steps=steps, rounds=a.rounds, result=result)) + "\n")


if __name__ == "__main__":
    main()
