"""Plays the newest model at normal speed, but only while someone is connected to watch."""
import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from torch.distributions import Distribution

import scoreboard as sb
from skill_env import SkillEnv, pace

ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument("--name", default="skill8")
p.add_argument("--port", type=int, default=25601)
a = p.parse_args()
out = ROOT / "runs" / "models" / a.name
torch.set_num_threads(1)
Distribution.set_default_validate_args(False)

env = SkillEnv(a.port)
env.realtime = True  # watchers see the full 4 seconds when it dies
env.c.call(op="release")
model, mtime = None, 0.0
wins, games = 0, 0


def status():
    try:
        return json.loads((out / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def watched():
    return env.c.call(op="watchers")["watchers"] > 0


def overlay():
    s = status()
    stages = s.get("stages", env.stage + 1)
    rate = wins / games if games else 0.0
    env.sidebar([["BRIDGE AI ", "white", True], ["LIVE", "gold", True]], [
        sb.stage_bar(env.stage + 1, stages),
        [["Gap ", sb.LABEL], [str(env.gap), sb.VALUE], ["  Height ", sb.LABEL], [f"{env.dy:+d}", sb.VALUE]],
        [["Task ", sb.LABEL], [sb.extras_text(env.kind or env.extras) or "plain bridge", "light_purple" if env.kind or env.extras else sb.VALUE]],
        sb.BLANK,
        sb.stat("Blocks left", f"{env.obs['held_count']}/{env.blocks_given}", sb.VALUE),
        sb.stat("Here", f"{wins}/{games}", sb.higher_is_better(rate, 0.85, 0.6) if games else sb.VALUE, "goals"),
        sb.stat("Test", f"{s.get('eval_success', 0):.0%}", sb.higher_is_better(s.get("eval_success", 0), 0.85, 0.6)),
        sb.stat("Speed", f"{s.get('speed', 0):.2f}", sb.higher_is_better(s.get("speed", 0), 2.5, 1.7), "blocks/s"),
        sb.BLANK,
        sb.stat("Model", sb.compact(s.get("steps", 0)), sb.VALUE, "steps"),
    ])


def episode():
    """Plays one episode. Returns None when the last watcher left partway through."""
    obs, _ = env.reset()
    overlay()
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
            overlay()
        next_t = pace(next_t, 0.1)
    return info


playing = False
while True:
    if not watched():
        if playing:
            env.c.call(op="release")
            playing = False
            print("nobody watching: idle", flush=True)
        time.sleep(1.0)
        continue
    if not (out / "latest.zip").exists():
        time.sleep(1.0)
        continue
    if not playing:
        playing = True
        print("watcher joined: playing", flush=True)
    m = (out / "latest.zip").stat().st_mtime
    if m != mtime:
        model = PPO.load(out / "latest.zip", device="cpu")
        mtime = m
        wins, games = 0, 0
    env.set_stage(status().get("stage", 0))
    info = episode()
    if info is None:
        continue
    games += 1
    wins += bool(info["success"])
    print(f"episode {games}: {'goal' if info['success'] else 'miss'} ({wins}/{games})", flush=True)
    time.sleep(0.7)
