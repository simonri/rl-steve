"""Two models fight each other directly: A's score against B (wins + half the draws) in each of the rating's kinds of
fight (rating.FIGHTS), with 95 % intervals, and over all of them. For A/B tests where the courses alone say too little.

    python tools/duel_pair.py A.zip B.zip [--rounds 128] [--label "ema vs raw"] [--out results.jsonl]
"""
import _path  # noqa: F401  (python/ on the import path)
import argparse
import json

import torch

import curriculum
import train_auto as ta
from benchmark import first_rounds, wilson
from opponents import Opponents
from rating import FIGHTS
from sim_env import SimVecEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--rounds", type=int, default=128)
    p.add_argument("--label", default="")
    p.add_argument("--out")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pa, pb = ta.load_policy(a.a, device), ta.load_policy(a.b, device)
    result, total = {}, 0.0
    for fight, dials in FIGHTS.items():
        levels = curriculum.Curriculum().levels
        levels["duel"].update(dials, bot=0)
        vec = SimVecEnv(a.rounds, levels, seed=616161)
        vec.env_method("set_exam", "duel", True)
        opp = Opponents(a.rounds)
        opp.policies["b"] = pb
        opp.choose = lambda meta: "b"
        infos = first_rounds(vec, pa, device, deterministic=False, opponents=opp)
        points = sum(1.0 if i["success"] else 0.0 if i["deaths"] > 0 else 0.5 for i in infos)
        lo, hi = wilson(points, len(infos))
        result[fight] = dict(score=round(points / len(infos), 3), low=round(lo, 3), high=round(hi, 3))
        total += points
    n = a.rounds * len(FIGHTS)
    lo, hi = wilson(total, n)
    result["all"] = dict(score=round(total / n, 3), low=round(lo, 3), high=round(hi, 3))
    line = f"{a.label or a.a + ' vs ' + a.b}: " + ", ".join(f"{f} {r['score']:.2f} [{r['low']:.2f}-{r['high']:.2f}]" for f, r in result.items())
    print(line, flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(dict(a=a.a, b=a.b, label=a.label, result=result)) + "\n")


if __name__ == "__main__":
    main()
