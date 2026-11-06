"""Real game check: how often the teacher clears the courses, with and without late keys. python test_planner.py <port> <rounds>"""
import _path  # noqa: F401  (python/ on the import path)
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from auto_env import AutoEnv
from curriculum import FAMILIES, Curriculum
from planner import Teacher

ROOT = Path(__file__).resolve().parent.parent.parent
env = AutoEnv(int(sys.argv[1]))
rounds = int(sys.argv[2])
teacher = Teacher(env)
saved = json.loads((ROOT / "runs/models/auto3/levels.json").read_text())


def play(family, chosen, late, label):
    levels = Curriculum(saved).levels
    levels[family].update({d: FAMILIES[family][d].index(v) for d, v in chosen.items()})
    env.set_levels(levels)
    env.set_exam(family)
    out = Counter()
    branches = Counter()
    for n in range(rounds):
        env._np_random = np.random.default_rng(3000 + n)
        env.reset()
        if late is not None:
            env.late_chance = env.aim_noise = late
        done, info = False, {}
        while not done:
            _, _, term, trunc, info = env.step(teacher.act())
            done = term or trunc
            if done and not info["clean"]:
                branches[teacher.branch] += 1
        out["clean" if info["clean"] else "won after death" if info["success"] else "stuck" if info["stalled"] else "two deaths" if env.deaths >= 2 else "out of time"] += 1
    print(f"{family:5s} {label:44s} late {late}: {dict(out)} last move when it failed {dict(branches)}", flush=True)
    return out["clean"] / rounds


hop = FAMILIES["hop"]
for late in (0.0, 0.3) if len(sys.argv) < 4 else (float(sys.argv[3]),):
    play("hop", dict(offset=(2, 0), count=4, height=(0,)), late, "2 ahead, flat, 4 jumps")
    play("hop", dict(offset=(3, 1), count=4, height=(-2, -1, 0, 1)), late, "3 ahead 1 side, heights, 4 jumps")
    play("hop", dict(offset=(2, 2), count=4, height=(0,)), late, "zigzag 2+2, 4 jumps")
    play("hop", dict(offset=(4, 2), count=4, height=(0,)), late, "4 ahead 2 side, 4 jumps")
    play("path", dict(length=14, wind="around", slope="stairs and drops", holes=True), late, "path 14, around, stairs and drops, holes")
env.c.close()
