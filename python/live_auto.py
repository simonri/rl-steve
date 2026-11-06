"""Plays the newest model at normal speed on every course kind, with all dials at their limit, while someone watches."""
import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from torch.distributions import Distribution

import scoreboard as sb
from auto_env import AutoEnv
from curriculum import FAMILIES, Curriculum, is_open
from skill_env import pace

ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument("--name", default="auto4")
p.add_argument("--port", type=int, default=25601)
a = p.parse_args()
out = ROOT / "runs" / "models" / a.name
torch.set_num_threads(1)
Distribution.set_default_validate_args(False)

env = AutoEnv(a.port)
env.realtime = True  # watchers see the full 4 seconds when it dies
env.c.call(op="release")
model, mtime = None, 0.0
score = {f: [0, 0] for f in FAMILIES}


def status():
    try:
        return json.loads((out / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def watched():
    return env.c.call(op="watchers")["watchers"] > 0


def overlay(s):
    lines = [[["Now ", sb.LABEL], [s.get("families", {}).get(env.family, {}).get("name", env.family), "light_purple"]]]
    if env.family == "hop":
        lines.append(sb.stat("Blocks", f"{env.landed}/{len(env.landing_order)}", sb.VALUE))
    elif env.blocks_given:
        lines.append(sb.stat("Blocks left", f"{env.obs['held_count']}/{env.blocks_given}", sb.VALUE))
    lines.append(sb.BLANK)
    for f, row in s.get("families", {}).items():
        if row["open"]:
            lines.append(sb.stat(row["name"], f"{row['level']}/{row['most']}", sb.VALUE, f"here {score[f][0]}/{score[f][1]}"))
    lines.append(sb.stat("Model", sb.compact(s.get("steps", 0)), sb.VALUE, "steps"))
    env.sidebar([["BRIDGE AI ", "white", True], ["LIVE", "gold", True]], lines)


def episode(s):
    """Plays one round. Returns None when the last watcher left partway through."""
    obs, _ = env.reset()
    overlay(s)
    next_t = time.perf_counter()
    done, decisions, info = False, 0, {}
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
        decisions += 1
        if decisions % 10 == 0:
            if not watched():
                return None
            overlay(s)
        next_t = pace(next_t, 0.05)
    return info


playing = False
turn = 0
while True:
    if not watched():
        if playing:
            env.c.call(op="release")
            playing = False
        time.sleep(1.0)
        continue
    if not (out / "latest.zip").exists():
        time.sleep(1.0)
        continue
    playing = True
    m = (out / "latest.zip").stat().st_mtime
    if m != mtime:
        model = PPO.load(out / "latest.zip", device="cpu")
        mtime = m
    s = status()
    levels = Curriculum(s.get("levels")).levels
    env.set_levels(levels)
    kinds = [f for f in FAMILIES if is_open(levels, f)]
    env.set_exam(kinds[turn % len(kinds)])
    turn += 1
    info = episode(s)
    if info is None:
        continue
    score[info["family"]][1] += 1
    score[info["family"]][0] += bool(info["clean"])
    print(info["family"], "clean" if info["clean"] else "miss", score[info["family"]], flush=True)
    time.sleep(0.7)
