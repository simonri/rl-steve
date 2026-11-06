"""Can one jump clear a given offset between two single blocks? Pure physics in the real game, no limits."""
import math
import sys

from mcclient import McClient
from skill_env import Y

c = McClient(int(sys.argv[1]))
OFFSETS = [(1, 2), (2, 2), (0, 3), (1, 3)]


def attempt(ox, oz, inset, run_ticks, let_go):
    """Starts inset from the corner nearest the target, runs a little if there is room, then jumps."""
    nx, nz = ox / math.hypot(ox, oz), oz / math.hypot(ox, oz)
    px = 0.5 + nx * inset
    pz = 0.5 + nz * inset
    yaw = math.degrees(math.atan2(-(ox + 0.5 - px), oz + 0.5 - pz))
    c.reset(clear=[-3, Y - 3, -3, 6, Y + 4, 6], platforms=[[0, Y, 0, 0, Y, 0], [ox, Y, oz, ox, Y, oz]], pos=[px, Y + 1, pz],
            yaw=yaw, pitch=0.0, blocks=0, block_radius=2)
    for _ in range(run_ticks):
        c.step(forward=True, sprint=True)
    o = c.step(forward=True, sprint=True, jump=True)
    for _ in range(40):
        x, y, z = o["pos"]
        remaining = math.hypot(ox + 0.5 - x, oz + 0.5 - z)
        if o["on_ground"] and remaining < 0.9 and abs(y - (Y + 1)) < 0.01:
            return True, remaining
        if y < Y:
            return False, remaining
        o = c.step(forward=remaining > let_go, sprint=remaining > let_go)
    return False, remaining


for ox, oz in OFFSETS:
    best = []
    for inset in (0.0, 0.3):
        for run_ticks in (0, 2, 4):
            for let_go in (0.1, 0.5):
                results = [attempt(ox, oz, inset, run_ticks, let_go)[0] for _ in range(3)]
                if any(results):
                    best.append((sum(results), inset, run_ticks, let_go))
    best.sort(reverse=True)
    good = sum(1 for b in best if b[0] == 3)
    print(f"offset {ox},{oz} distance {math.hypot(ox, oz):.2f}: {len(best)} of 12 settings land at least once, {good} land 3 of 3; best {best[:3]}", flush=True)
c.close()
print("ok")
