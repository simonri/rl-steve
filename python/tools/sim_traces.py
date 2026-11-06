"""Records what the real game does, tick by tick, as the answer key for the simulator.

Plays rounds through the normal AutoEnv on one client, with a trained model and some random inputs mixed in so that
falls, edges, placing and damage all show up. Every message to the game and every reply is written down: the keys,
click and turn sent, and the full state that came back. sim_check.py replays the same inputs in the simulator.

    python sim_traces.py --port 25601 --rounds 300 --out ../runs/sim_traces
"""
import _path  # noqa: F401  (python/ on the import path)
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Distribution

import curriculum
from auto_env import AutoEnv
from mcclient import McClient

ROOT = Path(__file__).resolve().parent.parent.parent


class Logged(McClient):
    """The normal client, writing down every call and reply."""

    def __init__(self, port, log):
        super().__init__(port)
        self.log = log

    def call(self, **msg):
        reply = super().call(**msg)
        self.log.append({"sent": msg, "got": reply})
        return reply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=25601)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--model", default=str(ROOT / "runs" / "models" / "auto5" / "latest.zip"))
    p.add_argument("--levels", default=str(ROOT / "runs" / "models" / "auto5" / "levels.json"))
    p.add_argument("--random", type=float, default=0.25, help="share of ticks with a random input instead of the model's")
    p.add_argument("--out", default=str(ROOT / "runs" / "sim_traces"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    Distribution.set_default_validate_args(False)
    torch.set_num_threads(1)
    from sb3_contrib import RecurrentPPO

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    env = AutoEnv.__new__(AutoEnv)
    real_client = McClient

    # AutoEnv builds its own client in __init__; hand it the logging one instead.
    import skill_env
    skill_env.McClient = lambda port: Logged(port, log)
    env.__init__(a.port, levels=json.loads(Path(a.levels).read_text()))
    skill_env.McClient = real_client
    model = RecurrentPPO.load(a.model, device="cpu")
    rng = np.random.default_rng(a.seed)
    env.reset(seed=a.seed)  # the first reset after start also wipes leftovers; not part of any trace
    families = ["hop", "path", "bridge", "bridge", "treenan"]
    written = 0
    t0 = time.time()
    for n in range(a.rounds):
        # Every dial at a random level up to the curriculum's, so easy and hard rounds both show up.
        levels = json.loads(Path(a.levels).read_text())
        family = families[n % len(families)]
        for d, values in curriculum.FAMILIES[family].items():
            levels[family][d] = int(rng.integers(0, max(levels[family][d], 1) + 1)) if d not in curriculum.EXPERT.get(family, ()) else 0
        env.set_levels(levels)
        env.set_exam(family)
        log.clear()
        obs, _ = env.reset()
        mem, first, done, info = None, True, False, {}
        while not done:
            act, mem = model.predict(obs, state=mem, episode_start=np.array([first]), deterministic=False)
            first = False
            if rng.random() < a.random:
                act = env.action_space.sample()
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        calls = list(log)  # from clearing and building this round on
        with gzip.open(out / f"trace{a.port}-{a.seed}-{n:04d}.json.gz", "wt") as f:
            json.dump({"family": family, "task": env.task, "levels": levels, "info": {k: v for k, v in info.items() if k != "parts"}, "calls": calls}, f)
        written += sum(1 for c in calls if c["sent"].get("op") == "step")
        if n % 10 == 0:
            print(f"{n + 1} rounds, {written} ticks, {written / (time.time() - t0):.0f} ticks/s, last {family} {'clean' if info.get('clean') else 'miss'}", flush=True)
    env.c.close()


if __name__ == "__main__":
    main()
