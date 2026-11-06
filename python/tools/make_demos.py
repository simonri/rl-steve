"""Plays random parkour and path rounds with the planning teacher and saves the clean ones, in the same format as recordings
from people, so the same copying step can use them. python make_demos.py <port> <rounds> <folder>"""
import _path  # noqa: F401  (python/ on the import path)
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

from auto_env import AutoEnv
from curriculum import FAMILIES
from planner import Teacher

KEYS = ("forward", "back", "left", "right", "sneak", "jump", "sprint")


def as_json(value):
    return list(value) if isinstance(value, tuple) else value


def main():
    port, rounds, folder = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    folder.mkdir(parents=True, exist_ok=True)
    env = AutoEnv(port)
    teacher = Teacher(env)
    rng = np.random.default_rng(port * 1000 + int(time.time()) % 1000)
    kept = tried = 0
    while tried < rounds:
        family = ("hop", "path")[int(rng.integers(2))]
        settings = {d: v[int(rng.integers(len(v)))] for d, v in FAMILIES[family].items()}
        env.set_exam(None)
        env._pick_stage = lambda family=family, settings=settings: (family, settings, None)
        try:
            env.reset()
        except RuntimeError as e:  # very long courses can reach past what the game has loaded yet
            print(f"{port}: skipped a round: {e}", flush=True)
            continue
        start = dict(meta=dict(family=family, settings={k: as_json(v) for k, v in settings.items()}, course=env.course, poles=env.poles,
                               landing_order=env.landing_order, goal=[env.goal_x, env.goal_z], dy=env.dy, par=env.par, fall_margin=env.fall_margin,
                               limit=env.limit, home=env.home["pos"], yaw=env.home["yaw"], pitch=env.home["pitch"], blocks_given=0, kind=None, blocked=[], detour=0),
                     player="teacher", started=time.time())
        rows, done, info, tick = [], False, {}, 0
        while not done:
            _, _, term, trunc, info = env.step(teacher.act())
            done = term or trunc
            tick += 1
            o = env.obs
            rows.append(dict({k: o[k] for k in ("pos", "vel", "yaw", "pitch", "on_ground", "health", "held_count", "feet", "solid", "surface", "hurt_time",
                                                 "sprinting", "collision", "fall_distance") if k in o},
                             **({"hit": o["hit"]} if o.get("hit") else {}), **{k: bool(env.keys[k]) for k in KEYS}, tick=tick,
                             event="won" if info.get("success") else ("died" if info.get("respawned") else None)))
        tried += 1
        if info.get("clean"):
            kept += 1
            path = folder / f"round{port}-{kept:05d}.jsonl.gz"
            with gzip.open(path, "wt") as f:
                f.write("\n".join(json.dumps(line) for line in (start, *rows, dict(result="won", deaths=0, ticks=tick, kept=len(rows)))) + "\n")
        if tried % 50 == 0:
            print(f"{port}: {kept}/{tried} clean", flush=True)
    print(f"{port}: done, {kept}/{tried} clean", flush=True)
    env.c.close()


if __name__ == "__main__":
    main()
