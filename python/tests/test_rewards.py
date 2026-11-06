"""Checks the point rules with a fake game. Values below are before the overall reward scale."""
import _path  # noqa: F401  (python/ on the import path)
import math

import skill_env
from skill_env import (AIM_REWARD, BLOCK_COST, DEATH_COST, EFFICIENT, EMPTY_CLICK_COST, FIRST_JUMP_BONUS, GOAL_REWARD, HEAD, JUMP, JUMP_SHIFT_COST, KNOWN_SLOT, LEARN,
                       MAX_DEATHS, RESPAWN_TICKS, SECURE, STAGES, STALL_COST, STALL_TICKS, STAYED_HOME_COST, TICK_COST, WASTE_COST, Y, SkillEnv)

SkillEnv.reward_scale = 1.0
G, AIR, VOID = Y + 1.0, Y + 2.0, Y - 3.0
stage_of = lambda phase, jump=False: next(i for i, t in enumerate(STAGES) if t[8] == phase and (t[6] == JUMP) == jump)
LEARN_BUILD, SECURE_BUILD, EFFICIENT_BUILD = stage_of(LEARN), stage_of(SECURE), stage_of(EFFICIENT)
LEARN_JUMP, SECURE_JUMP = stage_of(LEARN, True), stage_of(SECURE, True)


def state(x=0.5, y=G, ground=True, held=40, yaw=-90.0, vel=(0, 0, 0), health=20.0):
    return dict(pos=[x, y, 0.5], vel=list(vel), yaw=yaw, pitch=80.0, on_ground=ground, sneaking=False, held_count=held,
                health=health, solid="0" * 125)


class Fake:
    def __init__(self, path):
        self.path, self.calls, self.last = list(path), [], None

    def step(self, ticks=1, **a):
        self.calls.append(ticks)
        if ticks > 1:
            return self.last
        self.last = self.path.pop(0) if self.path else self.last
        return self.last

    def reset(self, **spec):
        assert spec["keep_arena"]
        return state(x=spec["pos"][0], held=spec["blocks"])


def make(stage, path, start=None, goal=20, limit=100000):
    env = SkillEnv.__new__(SkillEnv)
    env.c = Fake(path)
    env.mix = 0.0
    env.dirty = [0] * 6
    env.set_stage(stage)
    env.goal_x, env.goal_z, env.dy, env.par, env.limit = goal, 0, 0, 2, limit
    env.obs = start or state()
    env.home = dict(pos=[0.5, Y + 1, 0.5], yaw=0.0, pitch=0.0)
    env._begin_episode()
    env.late_chance = 0.0
    return env


def tick(env, use=False, jump=0, sneak=0):
    return env._tick(0, 0, sneak, use, jump, 0, 3, 3)


def near(a, b):
    assert abs(a - b) < 1e-6, (a, b)


# Scale: the same round pays a tenth with the real scale.
env = make(LEARN_BUILD, [state(x=1.5)])
SkillEnv.reward_scale = 0.1
near(tick(env)[0], (1.0 - TICK_COST) * 0.1)
assert abs(env.parts["progress"] - 0.1) < 1e-6
SkillEnv.reward_scale = 1.0

# Death costs more in later phases, and a jump into the void earns nothing back.
for phase, stage in ((LEARN, LEARN_BUILD), (SECURE, SECURE_BUILD), (EFFICIENT, EFFICIENT_BUILD)):
    env = make(stage, [state(x=2.5), state(x=5.5, y=AIR, ground=False), state(x=7.5, y=VOID, ground=False)])
    near(tick(env)[0], 2.0 - TICK_COST)
    near(tick(env)[0], -TICK_COST)
    r = tick(env)
    near(r[0], -TICK_COST - DEATH_COST[phase])
    assert r[3]["respawned"] and not r[1]
assert DEATH_COST[LEARN] < DEATH_COST[SECURE] < DEATH_COST[EFFICIENT]

# Blocks and speed count only in the efficient phase.
for stage, counted in ((LEARN_BUILD, False), (SECURE_BUILD, False), (EFFICIENT_BUILD, True)):
    env = make(stage, [state(held=39), state(held=38), state(held=37)], start=state(held=40))
    r1, r2, r3 = tick(env), tick(env), tick(env)
    near(r1[0], -TICK_COST - (BLOCK_COST if counted else 0))
    near(r3[0], -TICK_COST - (WASTE_COST if counted else 0))
    assert env.used == 3
env = make(EFFICIENT_BUILD, [state(x=20.5)], start=state(x=20.0), limit=1000)
near(tick(env)[3]["success"], True)
assert "speed" in env.parts and env.parts["goal"] == GOAL_REWARD
env = make(SECURE_BUILD, [state(x=20.5)], start=state(x=20.0), limit=1000)
tick(env)
assert "speed" not in env.parts and "block efficiency" not in env.parts

# Helpers exist only while learning a jump stage, and are paid at the edge toward the goal.
fly = [state(x=1.7, vel=(0.2, 0.42, 0), ground=False, y=AIR)]
env = make(LEARN_JUMP, list(fly), start=state(x=1.6, held=0))
near(tick(env, jump=1)[0], FIRST_JUMP_BONUS - TICK_COST)
env = make(SECURE_JUMP, list(fly), start=state(x=1.6, held=0))
near(tick(env, jump=1)[0], -TICK_COST)
env = make(LEARN_JUMP, [state(x=0.6, vel=(0.2, 0.42, 0), ground=False, y=AIR)], start=state(x=0.5, held=0))
near(tick(env, jump=1)[0], -TICK_COST)  # not at the edge
env = make(LEARN_JUMP, [state(yaw=0.0, held=0)], start=state(yaw=-90.0, held=0))
near(tick(env)[0], -TICK_COST + AIM_REWARD * (0.0 - 1.0))  # turning away gives points back

# Shift and clicking without blocks cost on the stages where they mean nothing.
env = make(LEARN_JUMP, [state(held=0)], start=state(held=0))
near(env._tick(0, 0, 1, False, 0, 0, 3, 3)[0], -TICK_COST - JUMP_SHIFT_COST)
env = make(SECURE_BUILD, [state(held=0)], start=state(held=0))
near(tick(env, use=True)[0], -TICK_COST - EMPTY_CLICK_COST)

# Being stuck costs, and more when it never left the start. The time limit is a real end of the round.
env = make(SECURE_BUILD, [state()] * 1)
env.since_best = STALL_TICKS - 1
r = tick(env)
near(r[0], -TICK_COST - STALL_COST - STAYED_HOME_COST)
assert r[1] and not r[2]
env = make(SECURE_BUILD, [state(x=5.5)], start=state(x=5.5))
env.since_best = STALL_TICKS - 1
near(tick(env)[0], -TICK_COST - STALL_COST)
env = make(SECURE_BUILD, [state(x=5.5)], start=state(x=5.5), limit=1)
r = tick(env)
assert r[1] and not r[2], "the time limit must end the round for real"

# Two deaths end the round. A respawn takes 4 seconds of game time, and old ground pays nothing.
env = make(SECURE_BUILD, [state(x=8.5), state(x=9.0, y=VOID, ground=False), state(x=1.5), state(x=9.5), state(x=9.0, y=VOID, ground=False)])
tick(env)
t = env.t
r = tick(env)
assert r[3]["respawned"] and env.t == t + 1 + RESPAWN_TICKS and RESPAWN_TICKS in env.c.calls
near(tick(env)[0], -TICK_COST)
near(tick(env)[0], 1.0 - TICK_COST)
r = tick(env)
assert MAX_DEATHS == 2 and r[1] and not r[3]["respawned"] and env.deaths == 2

# Stage plan: phases exist, gaps after the jumps cannot be jumped, budgets only tighten.
built = [t for i, t in enumerate(STAGES) if t[7] != (0.0, 0)]
assert all(a[7][0] >= b[7][0] for a, b in zip(built, built[1:]))
assert all(t[0] >= 5 for i, t in enumerate(STAGES) if i > skill_env.LAST_JUMP_STAGE and t[7] != (0.0, 0))
assert all(t[8] == EFFICIENT for t in STAGES[-5:])

# New inputs: known values are current even with a delayed view, and each new value works.
env = make(EFFICIENT_BUILD, [state()] * 10)
env.delay = 5
env.delayed = __import__("collections").deque(maxlen=6)
env.delayed.append(env._vec(env.obs))
env.limit, env.t, env.since_best, env.deaths = 100, 25, 10, 1
env.keys["sprint"], env.keys["jump"] = True, False
seen = env._with_timer(__import__("numpy").concatenate([env.delayed[0], *env.history]))
assert len(seen) == SkillEnv.N_OBS and HEAD == 35
near(seen[KNOWN_SLOT], 0.75)
near(seen[KNOWN_SLOT + 1], 0.75)
near(seen[KNOWN_SLOT + 2], 0.5)
near(seen[KNOWN_SLOT + 3], 1.0)
near(seen[KNOWN_SLOT + 4], 0.0)
env = make(EFFICIENT_BUILD, [])
floor = list("0" * 125)
for i in range(125):
    x, y, z = i // 25 - 2, (i // 5) % 5 - 2, i % 5 - 2
    if y == -1 and x <= 0:
        floor[i] = "1"
o = state(x=0.3)
o["solid"], o["feet"] = "".join(floor), [0, 101, 0]
near(env._edge_left(o), 0.35)  # the ground ends 0.7 blocks ahead, on a scale where 2 blocks is 1
o["hit"] = dict(type="block", block=[5, 100, 0], face="east")
assert env._place_gain(o) > 0.9
o["hit"]["face"] = "west"
assert env._place_gain(o) < -0.9
o["hit"]["face"] = "up"
near(env._place_gain(o), 0.0)
near(env._look_ahead(state(yaw=-90.0))[0], 1.0)

# Per-rule points on the dashboard carry their real sign and add up to the reward.
env = make(SECURE_BUILD, [state(x=1.5), state(x=2.0, y=VOID, ground=False)])
total = tick(env)[0] + tick(env)[0]
assert env.parts["death"] < 0 and env.parts["time"] < 0 and env.parts["progress"] > 0, env.parts
near(sum(env.parts.values()), total)

# The time limit follows the blocks the task takes.
env = make(SECURE_BUILD, [])
env.ticks_per_block, env.dy = 10, 0
env.par = env.blocks_needed(0, 16, 0)
near(env.time_limit(), 60 + 10 * 13)
env.par = env.blocks_needed(16, 16, 0)
near(env.time_limit(), 60 + 10 * 26)  # a diagonal takes about twice the blocks (and one for the corner), and gets the time for it
env.par = env.blocks_needed(0, 16, 0) + 4
near(env.time_limit(), 60 + 10 * 17)  # a detour around a wall
env.par, env.dy = env.blocks_needed(0, 16, -8), -8
near(env.time_limit(), 60 + 10 * 21)  # a drop

# Winding paths: one block wide, connected, holes only 1 or 2 long with ground on both sides, route follows them.
from skill_env import RUN, RUN_HOLES
for stage_kind, with_holes in ((RUN, False), (RUN_HOLES, True)):
    stage = next(i for i, t in enumerate(STAGES) if t[6] == stage_kind)
    for seed in range(40):
        env = make(stage, [])
        env.reset_seed = seed
        env._np_random = __import__("numpy").random.default_rng(seed)
        ux, uz = [(1, 0), (0, 1), (-1, 0), (0, -1)][seed % 4]
        path, holes, gx, gz = env._make_path(ux, uz, with_holes)
        assert len(path) + len(holes) >= env.gap - 2
        assert with_holes or not holes
        solid = set(path)
        for hole in holes:
            assert hole not in solid
        if with_holes:
            cells = sorted(holes)
            for h in holes:
                run = sum((h[0] + ux * k, h[1] + uz * k) in holes for k in range(-2, 3))
                assert run <= 2, (seed, holes)
        env.goal_x, env.goal_z = gx, gz
        env.blocked = env._off_path(path, holes)
        env._build_field()
        assert not (set(path) & env.blocked)
        x, z = 0.5, 0.5
        for _ in range(600):
            rx, rz = env._route(x, z)
            x, z = x + rx * 0.2, z + rz * 0.2
            if env._walk(x, z) < 0.5:
                break
        assert env._walk(x, z) < 0.5, (stage_kind, seed, x, z)

# Stopping on a block pays once per block. Height varies a little and is never a staircase.
from skill_env import ISLANDS, STOP_BONUS
stage = next(i for i, t in enumerate(STAGES) if t[6] == ISLANDS)
for seed in range(60):
    env = make(stage, [])
    env._np_random = __import__("numpy").random.default_rng(seed)
    course, holes, gx, gz, dy = env._make_course(1, 0, ISLANDS)
    heights = [Y] + [c[1] for c in course]
    changes = [b - a for a, b in zip(heights, heights[1:])]
    assert all(abs(c) <= 1 for c in changes) and all(abs(h - Y) <= 2 for h in heights), heights
    assert not any(a == b != 0 for a, b in zip(changes, changes[1:])), heights
env = make(stage, [state(x=3.5, y=Y + 1.0), state(x=3.5, y=Y + 1.0), state(x=3.5, y=Y + 1.0, vel=(0.2, 0, 0))], start=state(x=3.5, held=0))
env.stop_cells = {(3, Y + 1, 0)}
assert env._standing_on(state(x=3.1, y=Y + 1.0)) == (3, Y + 1, 0)
assert env._standing_on(state(x=2.8, y=Y + 1.0)) == (3, Y + 1, 0), "partly over the edge still counts"
env.goal_x, env.goal_z = 20, 0
env.ground_distance = env._distance(state(x=3.5, y=Y + 1.0))
first, second = tick(env)[0], tick(env)[0]
near(first, STOP_BONUS - TICK_COST)
near(second, -TICK_COST)

# Landing pays once per block and only in order, so skipping one earns nothing after it, respawned or not.
from skill_env import HOPS, LANDING_BONUS
stage = next(i for i, t in enumerate(STAGES) if t[6] == HOPS)
b1, b2, b3 = (3, Y + 1, 0), (5, Y + 1, 0), (7, Y + 1, 0)
on = lambda c, **k: state(x=c[0] + 0.5, y=c[1] + 0.0, held=0, **k)


def landings(path):
    env = make(stage, list(path), start=state(x=0.5, held=0))
    env.landing_order = [b1, b2, b3]
    env.ground_distance, env.best = 0.0, -100.0
    env._distance = lambda o: 0.0
    return [tick(env)[3] and env.parts.get("landed on a block", 0.0) for _ in path]


assert landings([on(b1), on(b1), on(b2), on(b3)]) == [LANDING_BONUS, LANDING_BONUS, 2 * LANDING_BONUS, 3 * LANDING_BONUS]
assert landings([on(b2), on(b3)]) == [0.0, 0.0], "skipped the first block"
assert landings([on(b1), on(b3), on(b2)]) == [LANDING_BONUS, LANDING_BONUS, 2 * LANDING_BONUS]
assert landings([state(x=b1[0] + 0.5, y=b1[1] + 0.5, ground=False, held=0)]) == [0.0], "in the air is not a landing"
assert landings([state(x=b1[0] - 0.2, y=b1[1] + 0.0, held=0)]) == [LANDING_BONUS], "on the edge of the block counts"

# The jump math agrees with what was measured in the game, and every course only asks for possible jumps.
from jump_physics import edge_gap, hits_pole, jump_possible, jump_reach
near(edge_gap(2, 2), math.sqrt(2))
for run_up, sprint, measured in ((0, True, 2.88), (5, True, 3.59), (10, True, 3.63), (0, False, 1.46), (10, False, 2.03)):
    assert abs(jump_reach(run_up, sprint=sprint) - measured) < 0.06, (run_up, sprint, jump_reach(run_up, sprint=sprint), measured)
assert jump_possible(4, 0, run_up_ticks=4, margin=0.0), jump_reach(4)  # gap 3 straight: measured 10/10
assert not jump_possible(6, 0, run_up_ticks=8, margin=0.0), jump_reach(8)  # gap 5: measured 0/10
assert jump_possible(2, 0) and jump_possible(2, 2) and jump_possible(3, 0), (jump_reach(0),)  # measured landing 3 of 3
for stage_kind, grid in (("climb", 1), ("hops", 0), ("islands", 0), ("wide_zigzag", 0)):
    stage = next(i for i, t in enumerate(STAGES) if t[6] == stage_kind)
    for seed in range(40):
        env = make(stage, [])
        env._np_random = __import__("numpy").random.default_rng(seed)
        course, holes, gx, gz, dy = env._make_course(1, 0, stage_kind)
        for a, b in zip(course, course[1:]):
            assert jump_possible(b[0] - a[0], b[2] - a[2], b[1] - a[1]), (stage_kind, a, b)
        landing = (gx - 2, gz)
        assert jump_possible(landing[0] - course[-1][0], landing[1] - course[-1][2], dy - (course[-1][1] - Y)), (stage_kind, course[-1], landing)

# Zigzag blocks change side every time, so a straight run cannot reach them.
from skill_env import WIDE_ZIGZAG
for seed, kind in [(s, WIDE_ZIGZAG) for s in range(20)]:
    stage = next(i for i, t in enumerate(STAGES) if t[6] == kind)
    env = make(stage, [])
    env._np_random = __import__("numpy").random.default_rng(seed)
    ux, uz = [(1, 0), (0, 1), (-1, 0), (0, -1)][seed % 4]
    course, holes, gx, gz, dy = env._make_course(ux, uz, kind)
    steps = [(abs((a[0] - b[0]) * ux + (a[2] - b[2]) * uz), abs((a[0] - b[0]) * uz) + abs((a[2] - b[2]) * ux)) for a, b in zip(course, course[1:])]
    assert set(steps) == {(2, 2)}, steps  # 2 ahead and 2 sideways: no straight line touches both rows
    # Every jump clears the poles, while skipping a block along one row hits one.
    centre = lambda c: (c[0] + 0.5, c[2] + 0.5)
    assert len(env.poles) == len(course), (env.poles, course)
    for a, b in zip(course, course[1:]):
        assert not hits_pole(centre(a), centre(b), env.poles), (a, b, env.poles)
    for a, b in zip(course, course[2:]):
        assert hits_pole(centre(a), centre(b), env.poles), ("skipping must be blocked", a, b, env.poles)
    sides = [c[2] if ux else c[0] for c in course]
    assert all(a != b for a, b in zip(sides, sides[1:])), sides
    assert dy == 0 and len(course) == env.gap
    env.goal_x, env.goal_z, env.dy = gx, gz, dy
    env.blocked = env._off_path([(c[0], c[2]) for c in course], holes)
    env._build_field()
    env.course, env.course_axis = course, (ux, uz)
    # Standing on a block, the way to go points at the next block.
    for a, b in zip(course, course[1:]):
        assert env._next_block(a[0] + 0.5, a[2] + 0.5) == (b[0] + 0.5, b[2] + 0.5)
    assert env._next_block(course[-1][0] + 0.5, course[-1][2] + 0.5) == (gx + 0.5, gz + 0.5)
    assert env._next_block(0.5, 0.5) == (course[0][0] + 0.5, course[0][2] + 0.5)

# Climbing goes up one block per step, with no bigger gap than allowed, and every block is on the route.
from skill_env import CLIMB, CLIMB_GAP, HOP_GAP, HOPS
for stage_kind, allowed in ((CLIMB, CLIMB_GAP), (HOPS, HOP_GAP)):
    stage = next(i for i, t in enumerate(STAGES) if t[6] == stage_kind)
    for seed in range(40):
        env = make(stage, [])
        env._np_random = __import__("numpy").random.default_rng(seed)
        ux, uz = [(1, 0), (0, 1), (-1, 0), (0, -1)][seed % 4]
        course, holes, gx, gz, dy = env._make_course(ux, uz, stage_kind)
        assert len(course) == env.gap
        last = (0, Y, 0) if stage_kind == CLIMB else None
        heights = [c[1] for c in course]
        if stage_kind == CLIMB:
            assert heights == [Y + i + 1 for i in range(env.gap)] and dy == env.gap, heights
        else:
            assert set(heights) == {Y} and dy == 0
        for a, b in zip(course, course[1:]):
            along = abs(a[0] - b[0]) + abs(a[2] - b[2])
            assert along <= allowed[1] + 2, (a, b)
        env.goal_x, env.goal_z, env.dy = gx, gz, dy
        env.blocked = env._off_path([(c[0], c[2]) for c in course], holes)
        env._build_field()
        env.course = ()
        x, z = 0.5, 0.5
        for _ in range(600):
            rx, rz = env._route(x, z)
            x, z = x + rx * 0.2, z + rz * 0.2
            if env._walk(x, z) < 0.5:
                break
        assert env._walk(x, z) < 0.5, (stage_kind, seed, x, z, course, holes)

# Landing damage on the goal costs once, also when it arrives during the ticks spent waiting for it.
env = make(LEARN_BUILD, [state(x=20.5), state(x=20.5, health=17.0)], goal=20)
env.reached = lambda o: True
tick(env)
near(env.parts["fall damage"], -1.5)
near(env.damage, 3.0)
print("rewards ok")
