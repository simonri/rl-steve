"""Plays the newest model at normal speed on every course kind, with all dials at their limit, while someone watches."""
import argparse
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from torch.distributions import Distribution

import scoreboard as sb
from auto_env import AutoEnv
from curriculum import FAMILIES, Curriculum, is_open
from skill_env import pace

ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument("--name", default="auto5")
p.add_argument("--port", type=int, default=25601)
a = p.parse_args()
out = ROOT / "runs" / "models" / a.name
torch.set_num_threads(1)
Distribution.set_default_validate_args(False)

env = AutoEnv(a.port)
env.realtime = True  # watchers see the full 4 seconds when it dies
env.c.call(op="release")
model, mtime = None, 0.0
score = defaultdict(lambda: [0, 0])  # course kinds added to a run later count from zero instead of stopping the viewer


def status():
    try:
        return json.loads((out / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def watched():
    return env.c.call(op="watchers")["watchers"] > 0


def overlay(s):
    """What viewers see: what the agent is, the task it is on now, how far the curriculum has come, and how this
    viewing session has gone."""
    families = s.get("families", {})
    head = "yellow"
    # The game shows 15 rows at most: heading and task share one row, so there is room for a gap before Trained.
    lines = [[["PPO + LSTM", sb.LABEL]], sb.BLANK, [["Now playing ", head, True], [families.get(env.family, {}).get("name", env.family), "light_purple"]]]
    if env.family == "hop":
        lines.append(sb.stat("Platforms", f"{env.landed}/{len(env.landing_order)}"))
    elif env.blocks_given:
        lines.append(sb.stat("Blocks", f"{env.obs['held_count']} left"))
    lines += [sb.BLANK, [["Curriculum", head, True]]]
    for f, row in families.items():
        if row["open"]:
            done = row["level"] >= row["most"]
            lines.append(sb.stat(row["name"], "max" if done else f"{row['level']}/{row['most']}", "green" if done else sb.VALUE))
    lines += [sb.BLANK, sb.stat("Trained", sb.compact(s.get("steps", 0)), sb.VALUE, "steps")]
    won, played = sum(v[0] for v in score.values()), sum(v[1] for v in score.values())
    if played:  # all rounds shown since the viewer started, so it only ever grows
        lines.append(sb.stat("Solved", f"{won}/{played}", sb.VALUE, "rounds shown"))
    env.sidebar([["Minecraft RL Agent", "gold", True]], lines)


def episode(s):
    """Plays one round. Returns None when the last watcher left partway through."""
    obs, _ = env.reset()
    overlay(s)
    next_t = time.perf_counter()
    done, decisions, info = False, 0, {}
    memory, start = None, True  # newer models carry a memory through the round
    while not done:
        act, memory = model.predict(obs, state=memory, episode_start=np.array([start]), deterministic=True)
        start = False
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
        decisions += 1
        if decisions % 10 == 0:
            if not watched():
                return None
            overlay(s)
        next_t = pace(next_t, 0.05)
    return info


def load(path):
    """Newer runs have a memory and older ones do not, and each loads only with its own kind."""
    with zipfile.ZipFile(path) as z:
        memory = "lstm" in z.read("data").decode()
    return (RecurrentPPO if memory else PPO).load(path, device="cpu")


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
        model = load(out / "latest.zip")
        mtime = m
    s = status()
    levels = Curriculum(s.get("levels")).levels
    env.set_levels(levels)
    kinds = [f for f in FAMILIES if is_open(levels, f)]
    env.set_exam(kinds[turn % len(kinds)], all_dials=False)  # the mix of courses it trains on, not only the hardest
    turn += 1
    info = episode(s)
    if info is None:
        continue
    score[info["family"]][1] += 1
    score[info["family"]][0] += bool(info["clean"])
    print(info["family"], "clean" if info["clean"] else "miss", score[info["family"]], flush=True)
    time.sleep(0.7)
