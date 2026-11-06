"""Replays recorded rounds from the real game in the simulator and measures how far apart they are.

Two ways:
  one tick: before every tick the simulator is put where the game was, so each error is that tick's own
  free run: the simulator plays the whole round on its own from the start, so errors add up

    python sim_check.py ../runs/sim_traces            # all traces
    python sim_check.py ../runs/sim_traces --show 3    # print the first ticks that differ
"""
import _path  # noqa: F401  (python/ on the import path)
import argparse
import gzip
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import mcsim

KEYS = ("forward", "back", "left", "right", "jump", "sneak", "sprint")


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def world_for(calls):
    """The blocks of the round: the map pasted for it, then its platforms. Everything else starts as air."""
    boxes, cells = [], {}
    for c in calls:
        m = c["sent"]
        if m.get("op") == "paste":
            for x, y, z, i in m["blocks"]:
                cells[(x, y, z)] = m["palette"][i]
        if m.get("op") == "reset" and not m.get("keep_arena"):
            boxes = [list(map(int, b)) for b in m.get("platforms", [])]
    xs = [b[0] for b in boxes] + [b[3] for b in boxes] + [k[0] for k in cells] + [0]
    ys = [b[1] for b in boxes] + [b[4] for b in boxes] + [k[1] for k in cells] + [100]
    zs = [b[2] for b in boxes] + [b[5] for b in boxes] + [k[2] for k in cells] + [0]
    lo, hi = [min(xs) - 8, min(ys) - 8, min(zs) - 8], [max(xs) + 8, max(ys) + 8, max(zs) + 8]
    placed = [(x, y, z, s) for (x, y, z), s in cells.items() if s != "minecraft:air"]
    known = {(x, y, z): True for x, y, z, _ in placed}
    for b in boxes:
        known.update({(x, y, z): True for x in range(b[0], b[3] + 1) for y in range(b[1], b[4] + 1) for z in range(b[2], b[5] + 1)})
    return mcsim.Single(boxes, placed, lo, hi), known


def put(sim, o, keys, crouch):
    sim.set_state(o["pos"], o["vel"], o["yaw"], o["pitch"], o["on_ground"], o["held_count"], health=o["health"], fall_distance=o.get("fall_distance", 0.0),
                  sprinting=o.get("sprinting", False), hurt_time=o.get("hurt_time", 0), keys=keys, crouch_pose=crouch, collision=o.get("collision", False))


def sync_blocks(sim, o, known):
    """Makes the blocks around the player what the game reported: a block the server took back is gone again, and
    one the simulator missed is there. Cells the world already has right are left alone, so slabs stay slabs."""
    solid, feet = o["solid"], [int(v) for v in o["feet"]]
    size = round(len(solid) ** (1 / 3))
    r = size // 2
    for i, ch in enumerate(solid):
        x, rest = divmod(i, size * size)
        y, z = divmod(rest, size)
        cell = (feet[0] + x - r, feet[1] + y - r, feet[2] + z - r)
        there = ch == "1"
        if known.get(cell, False) != there:
            sim.set_block(*cell, "minecraft:stone" if there else "minecraft:air")
            known[cell] = there


def hit_key(o):
    h = o.get("hit") or {}
    if h.get("type") != "block":
        return None
    return tuple(int(v) for v in h["block"]), h["face"]


def compare(real, sim, errs, n):
    d = math.dist(real["pos"], sim["pos"])
    errs["pos"].append(d)
    errs["vel"].append(math.dist(real["vel"], sim["vel"]))
    for k in ("on_ground", "sprinting", "collision", "held_count", "hurt_time"):
        n[k] += real.get(k) == sim.get(k)
    n["health"] += abs(real["health"] - sim["health"]) < 1e-3
    n["hit"] += hit_key(real) == hit_key(sim)
    n["fall"] += abs(real.get("fall_distance", 0.0) - sim["fall_distance"]) < 1e-3
    n["yaw"] += abs(real["yaw"] - sim["yaw"]) < 1e-4 and abs(real["pitch"] - sim["pitch"]) < 1e-4
    n["ticks"] += 1
    return d


def replay(trace, show=0, free=False):
    calls = trace["calls"]
    sim, known = world_for(calls)
    errs, n = defaultdict(list), Counter()
    shown = []
    prev, last_keys, crouch = None, [False] * 7, False
    recent = deque(maxlen=4)
    for i, c in enumerate(calls):
        m, o = c["sent"], c["got"]
        if m.get("op") == "reset":
            prev, last_keys, crouch = o, [False] * 7, False
            put(sim, o, last_keys, crouch)
            continue
        if m.get("op") != "step" or prev is None:
            continue
        a = m["action"]
        keys = tuple(bool(a.get(k, False)) for k in KEYS)
        if not free:
            put(sim, prev, last_keys, crouch)
            sync_blocks(sim, prev, known)
        for _ in range(m.get("ticks", 1)):
            s = sim.tick(list(keys), bool(a.get("use", False)), float(a.get("yaw", 0.0)), float(a.get("pitch", 0.0)))
        # Keep the world the same as the game's: a block only the simulator placed goes again, one the game placed
        # is added. Whether the two agree on placing is counted in held_count.
        game_placed = o["held_count"] < prev["held_count"] and o.get("clicks")
        if not free:
            if s.get("placed") and not game_placed:
                sim.set_block(*s["placed"][0], "minecraft:air")
            if game_placed:
                t = o["clicks"][-1]
                sim.set_block(int(t[0]), int(t[1]), int(t[2]), "minecraft:stone")
        d = compare(o, s, errs, n)
        if show and len(shown) < show and (d > 1e-6 or o["on_ground"] != s["on_ground"] or hit_key(o) != hit_key(s) or o["held_count"] != s["held_count"]
                                           or o.get("sprinting") != s["sprinting"]):
            shown.append((i, a, prev, o, s, list(recent)))
        recent.append((a, o.get("sprinting"), o["on_ground"], o["held_count"], o.get("sneaking")))
        last_keys = list(keys)
        crouch = keys[5]
        prev = o
    return errs, n, shown


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--free", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    files = sorted(Path(a.folder).glob("trace*.json.gz"))
    if a.limit:
        files = files[: a.limit]
    total, counts, worst = defaultdict(list), Counter(), []
    by_family = defaultdict(list)
    for f in files:
        t = load(f)
        errs, n, shown = replay(t, a.show, a.free)
        for k, v in errs.items():
            total[k] += v
        counts.update(n)
        by_family[t["family"]] += errs["pos"]
        if errs["pos"]:
            worst.append((max(errs["pos"]), f.name))
        for i, act, prev, o, s, before in shown:
            fmt = lambda v: [round(x, 5) for x in v]
            short = lambda a: "".join(k[0].upper() if a.get(k) else "." for k in (*KEYS, "use")) + f" {a.get('yaw', 0)}/{a.get('pitch', 0)}"
            print(f"{f.name} call {i}")
            for b_a, b_sprint, b_ground, b_held, b_sneak in before:
                print(f"    earlier {short(b_a)} -> sprinting {b_sprint} ground {b_ground} held {b_held} sneaking {b_sneak}")
            print(f"    this    {short(act)}\n  before pos {fmt(prev['pos'])} vel {fmt(prev['vel'])} ground {prev['on_ground']} sprinting {prev.get('sprinting')}"
                  f"\n  game   pos {fmt(o['pos'])} vel {fmt(o['vel'])} ground {o['on_ground']} held {o['held_count']} sprinting {o.get('sprinting')} hit {hit_key(o)}"
                  f"\n  sim    pos {fmt(s['pos'])} vel {fmt(s['vel'])} ground {s['on_ground']} held {s['held_count']} sprinting {s['sprinting']} hit {hit_key(s)}")
    ticks = counts["ticks"]
    pos = sorted(total["pos"])
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")
    print(f"{len(files)} rounds, {ticks} ticks ({'free run' if a.free else 'one tick at a time'})")
    print(f"position error: exact (<1e-6) {sum(d < 1e-6 for d in pos) / max(len(pos), 1):.4f}, median {q(pos, 0.5):.2e}, 99% {q(pos, 0.99):.2e}, worst {pos[-1] if pos else 0:.3f}")
    print("same as the game:", {k: round(counts[k] / max(ticks, 1), 4) for k in ("on_ground", "sprinting", "collision", "held_count", "health", "hurt_time", "fall", "hit", "yaw")})
    for fam, v in by_family.items():
        print(f"  {fam}: exact {sum(d < 1e-6 for d in v) / max(len(v), 1):.4f} over {len(v)} ticks")
    print("worst rounds:", sorted(worst)[-5:])


if __name__ == "__main__":
    main()
