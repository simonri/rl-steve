"""Real game check: measures game speed in viewer mode before and right after a respawn. No screenshots."""
import sys
import time

from skill_env import SkillEnv, pace

env = SkillEnv(int(sys.argv[1]), stage=13)
env.realtime = True
env.limit_override = None
env.reset()
env.limit = 100000
t0, tick0 = time.perf_counter(), env.t
next_t = time.perf_counter()
windows, deaths_seen, dead_time = [], 0, []
while deaths_seen < 3:
    t_before = time.perf_counter()
    _, _, term, trunc, info = env.step([1, 0, 0, 0, 0, 1, 3, 3])
    next_t = pace(next_t, 0.1)
    if info["respawned"]:
        deaths_seen += 1
        dead_time.append(time.perf_counter() - t_before)
        # The second right after a respawn is where it used to race. Walk in place with shift so it stays alive.
        start, ticks = time.perf_counter(), env.t
        for _ in range(10):
            env.since_best = 0
            env.step([0, 0, 1, 0, 0, 0, 3, 3])
            next_t = pace(next_t, 0.1)
        windows.append((env.t - ticks) / (time.perf_counter() - start))
    if term or trunc:
        env.reset()
        env.limit = 100000
print("seconds spent on each death:", [round(d, 1) for d in dead_time])
print("ticks per second right after respawn:", [round(w, 1) for w in windows])
assert all(3.8 < d < 5.0 for d in dead_time), dead_time
assert all(w < 23 for w in windows), windows
env.c.close()
print("ok")
