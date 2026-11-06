"""Turns recorded rounds into what the model sees and what it could have pressed, the same way as in training."""
import argparse
import gzip
import json
from collections import deque
from pathlib import Path

import numpy as np

import mcsim

from auto_env import TURN_STEPS, AutoEnv
from skill_env import HISTORY, MIN_KEY_TICKS

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("forward", "back", "left", "right", "jump", "sneak")
IDLE_TICKS = 10  # half a second


def is_idle(row, before):
    """No keys, no movement and no turning: nothing worth learning from in this tick."""
    if any(row[k] for k in KEYS) or row["pos"] != before["pos"]:
        return False
    return row["yaw"] == before["yaw"] and row["pitch"] == before["pitch"]


def trim(rows):
    """Keeps the last try only, starts when it first presses a key or moves (looking around on the spot is not playing yet), and cuts out
    stops of IDLE_TICKS or more."""
    start = max((i + 1 for i, row in enumerate(rows) if row.get("event") == "died"), default=0)
    rows = rows[start:]
    moves = next((i for i, row in enumerate(rows) if any(row[k] for k in KEYS) or row["pos"] != rows[0]["pos"]), len(rows) - 1)
    rows = rows[max(moves, 0):]
    kept, waiting = [], []
    for i, row in enumerate(rows):
        if not row.get("event") and is_idle(row, rows[max(i - 1, 0)]):  # the tick that wins always stays
            waiting.append(row)
            continue
        # A short pause is part of playing. The first move of the round has no pause before it to keep.
        if kept and len(waiting) < IDLE_TICKS:
            kept += waiting
        waiting = []
        kept.append(row)
    return kept


def nearest(steps, change):
    return min(range(len(steps)), key=lambda i: abs(change - steps[i]))


def turn(now, before):
    return (now - before + 180.0) % 360.0 - 180.0


def load(path):
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    return rows[0], rows[1:-1], rows[-1]


def to_tuples(settings):
    return {k: tuple(v) if isinstance(v, list) else v for k, v in settings.items()}


def replay(path):
    """Returns observations, actions, and whether the round was won. One row per tick."""
    head, ticks, end = load(path)
    ticks = trim(ticks)  # rounds keep what the rules were when they were saved, and this also covers older ones
    m = head["meta"]
    env = AutoEnv.__new__(AutoEnv)
    env.c, env.mix, env.dirty, env._np_random = None, 0.0, [0] * 6, np.random.default_rng(0)
    env.levels = None
    settings = to_tuples(m["settings"])
    env._use_task((m["family"], settings, None))
    env.course = [tuple(c) for c in m["course"]]
    env.landing_order = [tuple(c) for c in m["landing_order"]]
    env.goal_x, env.goal_z = m["goal"]
    env.dy, env.par, env.fall_margin = m["dy"], m["par"], m["fall_margin"]
    env.blocks_given, env.limit = m.get("blocks_given", 0), m["limit"]
    env.detour = m.get("detour", 0)
    env.blocked = {tuple(c) for c in m.get("blocked", ())}
    env.kind = m.get("kind")
    env._build_field()
    env.keys = {k: False for k in ("forward", "back", "left", "right", "sneak", "jump", "sprint")}
    env.key_age = {k: MIN_KEY_TICKS for k in env.keys}
    env.t, env.used, env.deaths, env.since_best, env.sneak_held = 0, 0, end["deaths"], 0, 0
    env.best = float("inf")
    history = deque([np.zeros(8, np.float32)] * HISTORY, maxlen=HISTORY)
    delayed = deque(maxlen=1)  # the model sees the game as it is now
    env.history2 = deque([(0.0, 0.0)] * HISTORY, maxlen=HISTORY)  # recordings never fight: no clicks, the first slot
    observations, actions = [], []
    env.landed = 0
    env._release_inputs()
    env.visited = set()
    fallen = 0.0  # older recordings do not have this, so it is worked out from the path
    hurt, last_health = 0, None  # nor this
    for here, then in zip(ticks, ticks[1:]):
        o = dict(here)
        o.setdefault("held_count", 0)  # older recordings had no blocks at all
        fallen = 0.0 if o["on_ground"] else fallen + max(0.0, -o["vel"][1])
        o.setdefault("fall_distance", fallen)
        env._track_jump(o, here["jump"])
        env._visit(o)
        hurt = 10 if last_health is not None and o["health"] < last_health else max(0, hurt - 1)
        last_health = o["health"]
        o.setdefault("hurt_time", hurt)
        # Where it stands decides which block counts as next, also when it skipped some.
        if o["on_ground"]:
            for i in range(env.landed, len(env.landing_order)):
                if env._is_on(o, env.landing_order[i]):
                    env.landed = i + 1
                    break
        env.obs = o
        delayed.append(env._vec(o))
        env.since_best = 0 if env._distance(o) < env.best - 0.01 else env.since_best + 1
        env.best = min(env.best, env._distance(o))
        seen = env._with_timer(np.concatenate([delayed[0], *history, env._combat(o)]))
        move = 1 if then["forward"] and not then["back"] else 2 if then["back"] and not then["forward"] else 0
        strafe = 1 if then["left"] and not then["right"] else 2 if then["right"] and not then["left"] else 0
        placed = max(0, o["held_count"] - then.get("held_count", 0))
        env.used += placed
        action = [move, strafe, int(then["sneak"]), min(placed, 1), int(then["jump"]), int(then["sprint"]),
                  nearest(TURN_STEPS, turn(then["yaw"], here["yaw"])), nearest(TURN_STEPS, then["pitch"] - here["pitch"]), 0, 0]  # no left click, keep the slot
        observations.append(seen)
        actions.append(action)
        history.append(np.asarray([move / 2, strafe / 2, action[2], action[3] / 2, action[4], action[5], action[6] / (len(TURN_STEPS) - 1), action[7] / (len(TURN_STEPS) - 1)], np.float32))
        env.keys.update(forward=then["forward"], back=then["back"], left=then["left"], right=then["right"], sneak=then["sneak"], jump=then["jump"], sprint=then["sprint"])
        env.sneak_held = env.sneak_held + 1 if then["sneak"] else 0
        env.t += 1
    shape = (0, AutoEnv.N_OBS)
    return (np.asarray(observations, np.float32) if observations else np.zeros(shape, np.float32),
            np.asarray(actions, np.int64) if actions else np.zeros((0, mcsim.ACTIONS), np.int64), end["result"] == "won" and end["deaths"] == 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default=str(ROOT / "runs" / "demos"))
    p.add_argument("--out", default=str(ROOT / "runs" / "demos.npz"))
    a = p.parse_args()
    obs, act, won = [], [], 0
    files = sorted(Path(a.demos).glob("*/round*.jsonl.gz"))
    for path in files:
        o, x, w = replay(path)
        obs.append(o)
        act.append(x)
        won += bool(w)
    np.savez_compressed(a.out, obs=np.concatenate(obs), actions=np.concatenate(act))
    print(f"{len(files)} rounds, {won} clean, {sum(len(o) for o in obs)} ticks -> {a.out}")
