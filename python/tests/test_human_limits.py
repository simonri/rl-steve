"""Checks the human limits with a fake game that records exactly what would reach Minecraft."""
import _path  # noqa: F401  (python/ on the import path)
import numpy as np

from skill_env import SNEAK_SLOT, MAX_SNEAK_TICKS, ISLAND, SLALOM, WALL, SNEAK_CLICK_TICKS, HISTORY, KEYS, MIN_KEY_TICKS, Y, SkillEnv


class Recorder:
    def __init__(self):
        self.sent = []

    def step(self, **action):
        self.sent.append(action)
        i = len(self.sent)
        return dict(pos=[0.5, Y + 1.0, 0.5 + i * 0.01], vel=[0, 0, 0], yaw=180.0, pitch=80.0, on_ground=True, sneaking=False,
                    held_count=64, health=20.0, solid="0" * 125)


def env_at(stage):
    env = SkillEnv.__new__(SkillEnv)
    env.home = dict(pos=[0.5, Y + 1, 0.5], yaw=0.0, pitch=0.0)
    env.c = Recorder()
    env.mix = 0.0
    env.dirty = [0, 0, 0, 0, 0, 0]
    env.set_stage(stage)
    env.goal_x, env.goal_z, env.dy, env.par, env.limit = 30, 0, 0, 2, 100000
    env.obs = env.c.step()
    env.c.sent.clear()
    env.reset_rng = env.np_random
    env._begin_episode()
    return env


FULL = 22
rng = np.random.default_rng(0)
env = env_at(FULL)
assert (env.delay_low, env.delay_high, env.late_chance) == (1, 1, 0.3)
for _ in range(5000):
    # A deliberately jittery agent, changing every input at random every tick.
    env.since_best = 0
    env.step([rng.integers(3), rng.integers(3), rng.integers(2), rng.integers(3), rng.integers(2), rng.integers(2), rng.integers(7), rng.integers(7)])
sent = env.c.sent
for key in KEYS:
    runs, last, length = [], sent[0][key], 0
    for a in sent:
        if a[key] == last:
            length += 1
        else:
            runs.append(length)
            last, length = a[key], 1
    assert runs and min(runs[1:]) >= MIN_KEY_TICKS, (key, min(runs))
assert max(abs(a["yaw"]) for a in sent) <= 60.3
assert max(abs(a["pitch"]) for a in sent) <= 60.3
assert all(isinstance(a["use"], bool) for a in sent)
assert any(a["yaw"] not in (0.0, 1.0, -1.0, 10.0, -10.0, 60.0, -60.0) for a in sent), "no aim noise"

# A change wanted now must arrive now or one tick later, never later than that.
env = env_at(FULL)
env.late_chance = 0.3
delays = []
for _ in range(400):
    start = len(env.c.sent)
    env.since_best = 0
    want = not env.keys["sneak"]
    for t in range(6):
        env.since_best = 0
        env.step([0, 0, int(want), 0, 0, 0, 3, 3])
    arrived = next(i for i, a in enumerate(env.c.sent[start:]) if a["sneak"] == want)
    delays.append(arrived)
assert set(delays) <= {0, 1, 2} and 1 in delays and 0 in delays, set(delays)

# The agent sees the game late: what it gets back is the state from `delay` ticks ago.
env = env_at(FULL)
env.delay = 4
env.delayed = __import__("collections").deque(maxlen=5)
truth = []
real_vec = env._vec


def recording(o):
    v = real_vec(o)
    truth.append(v)
    return v


env._vec = recording
env._seen(env._vec(env.obs))
for _ in range(10):
    env.since_best = 0
    v, *_ = env.step([0, 0, 0, 0, 0, 0, 3, 3])
    if len(truth) > 5:
        # Everything is late except the values a player always knows: shift timer, time, deaths, keys.
        from skill_env import KNOWN_SLOT
        always_known = [SNEAK_SLOT, *range(KNOWN_SLOT, KNOWN_SLOT + 5)]
        late = np.delete(v[:len(truth[0])], always_known)
        assert np.array_equal(late, np.delete(truth[-5], always_known))
assert len(v) == SkillEnv.N_OBS and len(v) == len(truth[0]) + 8 * HISTORY

# Two clicks per decision reaches 20 per second, one gives 10.
env = env_at(FULL)
env.late_chance = 0.0
env.step([0, 0, 1, 0, 0, 0, 3, 3])
env.c.sent.clear()
for _ in range(50):
    env.since_best = 0
    env.step([0, 0, 1, 2, 0, 0, 3, 3])
assert sum(a["use"] for a in env.c.sent) == 100
env = env_at(FULL)
env.late_chance = 0.0
env.step([0, 0, 1, 0, 0, 0, 3, 3])
env.c.sent.clear()
for _ in range(50):
    env.since_best = 0
    env.step([0, 0, 1, 1, 0, 0, 3, 3])
assert sum(a["use"] for a in env.c.sent) == 50

# Moonbridge rule: clicks without shift never reach the game, right after a tap they do.
env = env_at(FULL)
env.late_chance = 0.0
for _ in range(50):
    env.since_best = 0
    env.step([2, 0, 0, 2, 0, 0, 3, 3])
assert sum(a["use"] for a in env.c.sent) == 0
env = env_at(FULL)
env.late_chance = 0.0
env.step([2, 0, 1, 0, 0, 0, 3, 3])
env.c.sent.clear()
env.key_age["sneak"] = 9
for _ in range(6):
    env.since_best = 0
    env.step([2, 0, 0, 2, 0, 0, 3, 3])
assert 1 <= sum(a["use"] for a in env.c.sent) <= SNEAK_CLICK_TICKS

# Shift lets go after the limit even if the agent keeps asking for it, and can then be pressed again.
env = env_at(FULL)
env.late_chance = 0.0
for _ in range(100):
    env.since_best = 0
    env.step([0, 0, 1, 0, 0, 0, 3, 3])
runs, length, offs = [], 0, []
for a in env.c.sent:
    if a["sneak"]:
        length += 1
    elif length:
        runs.append(length)
        length = 0
assert runs and max(runs) <= MAX_SNEAK_TICKS, runs
assert len(runs) >= 4, runs
offs, length = [], 0
for a in env.c.sent[runs[0]:]:
    if not a["sneak"]:
        length += 1
    elif length:
        offs.append(length)
        length = 0
assert offs and min(offs) >= MIN_KEY_TICKS, offs

# The shift timer is current even though the rest of the view is late.
env = env_at(FULL)
env.late_chance = 0.0
env.delay = 5
env.delayed = __import__("collections").deque(maxlen=6)
env._seen(env._vec(env.obs))
values = []
for _ in range(14):
    env.since_best = 0
    v, *_ = env.step([0, 0, 1, 0, 0, 0, 3, 3])
    values.append((round(float(v[SNEAK_SLOT]), 3), env.keys["sneak"], env.sneak_held))
for value, down, held in values:
    want = round(1.0 - 0.8 * held / MAX_SNEAK_TICKS, 3) if down else 0.0
    assert abs(value - want) < 0.002, values
assert values[0][0] > 0.85 and values[8][0] < 0.4, values
head = env._vec(env.obs)
assert head[SNEAK_SLOT] in (0.0, 1.0)

# Obstacles: the way around must be longer than the straight line, and never pass through stone.
for kind, ux, uz, gx, gz in ((WALL, 1.0, 0.0, 16, 0), (SLALOM, 0.0, 1.0, 0, 16), (WALL, 0.707, 0.707, 12, 12)):
    env = env_at(FULL)
    env.goal_x, env.goal_z, env.dy = gx, gz, 0
    boxes, env.blocked, env.detour = env._obstacles(kind, ux, uz, gx, gz, Y)
    env._build_field()
    assert env.blocked and env.detour > 0
    assert env._walk(0.5, 0.5) > __import__("math").hypot(gx, gz) - 3
    x, z = 0.5, 0.5
    for _ in range(200):
        rx, rz = env._route(x, z)
        x, z = x + rx * 0.25, z + rz * 0.25
        assert (__import__("math").floor(x), __import__("math").floor(z)) not in env.blocked, (kind, x, z)
        if env._walk(x, z) < 0.5:
            break
    assert env._walk(x, z) < 0.5, (kind, x, z, env._walk(x, z))
env = env_at(FULL)
env.goal_x, env.goal_z, env.dy = 16, 0, 0
boxes, blocked, detour = env._obstacles(ISLAND, 1.0, 0.0, 16, 0, Y)
assert not blocked and detour == -3 and len(boxes) == 1

# The viewer looks at the middle of the route, from far enough away to see both platforms.
for gx, gz, gy in ((16, 0, Y), (0, -16, Y - 8), (12, 12, Y + 3), (-20, 7, Y)):
    x, y, z, yaw, pitch = SkillEnv.camera_for(gx, gz, gy)
    m = __import__("math")
    mx, my, mz = gx / 2 + 0.5, (Y + 1 + gy + 1) / 2, gz / 2 + 0.5
    fx, fz = -m.sin(m.radians(yaw)), m.cos(m.radians(yaw))
    to = m.hypot(mx - x, mz - z)
    assert abs((mx - x) / to - fx) < 1e-6 and abs((mz - z) / to - fz) < 1e-6
    assert abs(m.degrees(m.atan2(y - my, to)) - pitch) < 1e-6
    half = m.degrees(m.atan2(m.hypot(gx, gz) / 2 + 2.5, m.hypot(to, y - my)))
    assert 25 < half < 50, half  # the user plays at FOV 90 vertically, which is wider still sideways

# Block budget: always enough for the estimate, tighter later, none on the jump stages.
from skill_env import JUMP, STAGES
factors = [t[7][0] for t in STAGES if t[7] != (0.0, 0)]
assert factors == sorted(factors, reverse=True) and min(factors) >= 1.5, factors
assert all(t[7] == (0.0, 0) for t in STAGES if t[6] == JUMP)
first_jump = next(i for i, t in enumerate(STAGES) if t[6] == JUMP)
assert first_jump == 4
env = env_at(5)
env.reset_rng = None
spec = env._spec()
ux, uz = (1 if env.goal_x > 0 else -1 if env.goal_x < 0 else 0), (1 if env.goal_z > 0 else -1 if env.goal_z < 0 else 0)
import math as _m
fx, fz = -_m.sin(_m.radians(spec["yaw"])), _m.cos(_m.radians(spec["yaw"]))
assert fx * ux + fz * uz > 0.95, "the first gap 2 jump stage should start facing the goal"
assert spec["blocks"] == 0

# Every building stage after the jumps needs a gap that cannot be jumped in any direction.
after_jumps = [t for i, t in enumerate(STAGES) if i > first_jump and t[7] != (0.0, 0)]
assert after_jumps and all(t[0] >= 5 for t in after_jumps), after_jumps

# Old jump rounds never appear on building stages, and old building rounds never on jump stages.
env = env_at(22)
env.mix = 1.0
assert not any(STAGES[env._pick_stage()][7] == (0.0, 0) for _ in range(300))
jump_at = max(i for i, t in enumerate(STAGES) if t[6] == JUMP)
env = env_at(jump_at)
env.mix = 1.0
assert all(STAGES[env._pick_stage()][6] == JUMP for _ in range(100))
print("human limits ok")
