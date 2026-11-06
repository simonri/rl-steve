"""Real game check, no screenshots: a watcher must be moved correctly for every episode, even very short ones."""
import _path  # noqa: F401  (python/ on the import path)
import sys
import time

from skill_env import SkillEnv

env = SkillEnv(int(sys.argv[1]))
env.c.call(op="release")
deadline = time.time() + 100
while env.c.call(op="watchers")["watchers"] < 1:
    assert time.time() < deadline, "no watcher joined"
    time.sleep(1)
env.set_stage(16)
bad, checked = 0, 0
for episode in range(40):
    spec = env._spec()
    env.c.reset(**spec)
    # Half the episodes end at once, like a quick fall. The rest run for a second of game time at normal speed.
    ticks = 1 if episode % 2 else 20
    for _ in range(ticks):
        env.c.step()
        time.sleep(0.05)
    w = env.c.call(op="watchers")["positions"][0]
    cam = spec["camera"]
    position = max(abs(w[i] - cam[i]) for i in range(3))
    turn = (w[3] - cam[3] + 180) % 360 - 180
    look = abs(w[4] - cam[4])
    checked += 1
    if position > 0.2 or abs(turn) > 1 or look > 1:
        bad += 1
        print(f"episode {episode} off: position {position:.2f}, yaw {turn:.1f}, pitch {look:.1f}", flush=True)
print(f"{checked - bad}/{checked} placements correct", flush=True)
env.c.close()
sys.exit(1 if bad else 0)
