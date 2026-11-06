"""Real game check: a drop must cost health, and standing still must end the episode after 2 s."""
import _path  # noqa: F401  (python/ on the import path)
import sys

from skill_env import SkillEnv, Y

env = SkillEnv(int(sys.argv[1]))
env.reset(seed=1)
NOOP = [0, 0, 0, 0, 0, 0, 3, 3]
ticks, info = 0, {}
while True:
    _, r, term, trunc, info = env.step(NOOP)
    ticks += 1
    if term or trunc:
        break
print(f"standing still: ended after {ticks} ticks, stalled={info['stalled']}, last reward {r:.2f}")

for drop, want_damage in ((3, False), (8, True)):
    o = env.c.reset(clear=[-8, Y - 16, -8, 12, Y + 8, 12], platforms=[[-1, Y, -1, 1, Y, 1], [2, Y - drop, -2, 8, Y - drop, 2]],
                    pos=[0.5, Y + 1, 0.5], yaw=270.0, pitch=0.0, blocks=64, block_radius=2)
    start = o["health"]
    low = start
    for _ in range(120):
        o = env.c.step(forward=True)
        low = min(low, o["health"])
    print(f"drop {drop}: health {start} -> {low}, y {o['pos'][1]}, expected damage: {want_damage}")
    assert (low < start) == want_damage
env.c.close()
print("ok")
