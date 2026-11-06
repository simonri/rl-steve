"""Real game check: inputs we punish as having no effect must leave the trajectory unchanged."""
import _path  # noqa: F401  (python/ on the import path)
import sys

from mcclient import McClient
from skill_env import Y

c = McClient(int(sys.argv[1]))


def run(start_pitch, ticks=60, **extra):
    c.reset(clear=[-8, Y - 8, -8, 8, Y + 8, 24], platforms=[[-3, Y, -3, 3, Y, 20]], pos=[0.5, Y + 1, 0.5],
            yaw=180.0, pitch=start_pitch, blocks=64, block_radius=2)
    out = []
    for _ in range(ticks):
        o = c.step(back=True, **extra)
        out.append((o["pos"], o["vel"], o["pitch"], o["held_count"]))
    return out


base = run(80.0)
assert run(80.0) == base, "baseline is not repeatable"
assert run(80.0, sprint=True) == base, "sprint without forward changed something"
print("sprint without forward: no effect")
sky = run(-90.0)
assert run(-90.0, use=True) == sky, "use with nothing to click changed something"
print("use without a target: no effect")
limited = run(-90.0, pitch=-5.0)
assert limited == sky, "looking past the limit changed something"
print("looking past the limit: no effect")
c.close()
print("ok")
