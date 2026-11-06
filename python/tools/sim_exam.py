"""Where a model fails: plays exam rounds of one course kind in the simulator and sorts how each one went.

    python sim_exam.py --name auto6 --family hop --rounds 256
"""
import _path  # noqa: F401  (python/ on the import path)
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from torch.distributions import Distribution

import curriculum
from sim_env import SimVecEnv

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="auto6")
    p.add_argument("--family", default="hop")
    p.add_argument("--rounds", type=int, default=256)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--random", action="store_true", help="with the randomness it trains with")
    a = p.parse_args()
    Distribution.set_default_validate_args(False)
    torch.set_num_threads(4)
    out = ROOT / "runs" / "models" / a.name
    levels = json.loads((out / "levels.json").read_text())
    print("dials:", {d: f"{levels[a.family][d]}/{len(v) - 1} ({v[levels[a.family][d]]})" for d, v in curriculum.FAMILIES[a.family].items()})
    model = RecurrentPPO.load(out / "latest.zip", device="cpu")
    vec = SimVecEnv(a.envs, levels, seed=7)
    vec.env_method("set_exam", a.family)
    specs = {}
    original = vec._new

    def new(i):  # remember each round's course, to say which jump went wrong
        original(i)
        m = vec.makers[i]
        specs[i] = dict(landing=list(m.landing_order), limit=m.limit, course=list(m.course), task=dict(m.task))

    vec._new = new
    obs = vec.reset()
    state, starts = None, np.ones(a.envs, bool)
    done_rounds, outcomes, failed_jump, details = 0, Counter(), Counter(), []
    tried, made, stuck_at = Counter(), Counter(), Counter()
    while done_rounds < a.rounds:
        act, state = model.predict(obs, state=state, episode_start=starts, deterministic=not a.random)
        spec_before = dict(specs)
        obs, _, dones, infos = vec.step(act)
        starts = dones
        for i in np.flatnonzero(dones):
            info, s = infos[i], spec_before[i]
            done_rounds += 1
            n = len(s["landing"])
            if info["clean"]:
                kind = "clean"
            elif info["stalled"]:
                kind = "stuck"
            elif info["deaths"] >= 2 or info["deaths"] and not info["success"]:
                kind = "died twice" if info["deaths"] >= 2 else "died, then ran out of time"
            elif info["success"]:
                kind = "won after a death"
            else:
                kind = "out of time"
            outcomes[kind] += 1
            for k in range(min(info["landed"] + (not info["clean"]), n)):
                to = s["course"][k]
                fr = s["course"][k - 1] if k else (0, 100, 0)
                ahead = (abs(to[0] - fr[0]), abs(to[2] - fr[2]))
                kind_of_jump = (f"gap {max(ahead) - 1} x {min(ahead)}", f"up {to[1] - fr[1]}", "first" if k == 0 else "later")
                tried[kind_of_jump] += 1
                made[kind_of_jump] += k < info["landed"]
                if k == info["landed"] and kind == "stuck":
                    stuck_at[kind_of_jump] += 1
            if not info["clean"] and info["landed"] < n:
                # The jump it did not make: from the block before (or the start) to the next one.
                k = info["landed"]
                to = s["course"][k]
                fr = s["course"][k - 1] if k else (0, 100, 0)
                ahead = (abs(to[0] - fr[0]), abs(to[2] - fr[2]))
                failed_jump[(f"jump {k + 1} of {n}", f"gap {max(ahead) - 1} x {min(ahead)}", f"up {to[1] - fr[1]}")] += 1
            details.append((kind, info["landed"], n, info["ticks"], s["limit"]))
    print(f"{done_rounds} {a.family} exam rounds ({'with' if a.random else 'without'} randomness):")
    for k, v in outcomes.most_common():
        print(f"  {v:4d} {k}")
    print("jumps it did not make (which jump, blocks between ahead x sideways, step up):")
    for k, v in failed_jump.most_common(15):
        print(f"  {v:4d} {k}")
    print("each kind of jump: made / tried (stuck before it), worst first:")
    rows = sorted(tried, key=lambda k: made[k] / tried[k])
    for k in rows:
        if tried[k] >= 5:
            print(f"  {made[k] / tried[k]:5.0%} {made[k]:4d}/{tried[k]:<4d} stuck {stuck_at[k]:3d}  {k}")
    timeouts = [d for d in details if d[0] == "out of time"]
    if timeouts:
        print(f"out of time: landed {np.mean([d[1] / d[2] for d in timeouts]):.0%} of the blocks on average, limit {np.mean([d[4] for d in timeouts]):.0f} ticks")


if __name__ == "__main__":
    main()
