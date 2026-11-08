"""Bridge to a goal in any direction and height, as fast as possible.

Everything horizontal is expressed in a frame turned by the nearest 90 degrees of the player's yaw,
so a bridge going north looks the same to the model as one going east.
"""
import heapq
import math
import time
from collections import deque

import gymnasium as gym
import numpy as np

from mcclient import McClient
from treenan import MAX_GAP, MIN_GAP, Treenan

Y = 100
R = 2
YAW_STEPS = (-60.0, -10.0, -1.0, 0.0, 1.0, 10.0, 60.0)  # degrees per tick, at most 1200 degrees per second
PITCH_STEPS = YAW_STEPS
FLAT, HILLS, DROPS = (0,), (-3, -2, -1, 0, 1, 2, 3), (-8, -6, -4, -3, -1, 0, 1, 3)
BLOCK_COST = 0.02  # per block up to the estimated need
WASTE_COST = 0.3  # per block beyond it
EFFICIENCY_BONUS = 5.0  # paid at the goal, scaled by how close to the estimate the agent stayed
UNUSED_BLOCK_COST = 0.3  # per block that turned out to do nothing, paid at the goal
NO_EFFECT_COST = 0.002  # per input that provably changed nothing
FACES = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
# Human limits. Timing tricks that need exact ticks stop working here, without naming any technique.
KNOWN_SLOT = 25  # first of the values that are never delayed
SNEAK_SLOT = 15  # position of the sneak value in the observation
MAX_SNEAK_TICKS = 20  # shift lets go by itself after 1 second, so holding it all the way is not an option
SNEAK_CLICK_TICKS = 2  # a click only counts with shift down now or this recently, which rules out moonbridging
TICKS_PER_DECISION = 2  # 10 decisions per second
HISTORY = 5  # recent inputs in the observation, since the agent sees the game late
MIN_KEY_TICKS = 2  # a key stays pressed or released for at least 100 ms
AIM_NOISE_FULL = 0.3  # degrees of noise on each camera movement
# From this stage index on: (reaction ticks low, high, chance an input arrives late, aim noise in degrees)
HUMAN_LEVELS = (
    (0, (1, 1, 0.1, 0.1)),
    (3, (1, 1, 0.2, 0.2)),
    (15, (1, 1, 0.3, AIM_NOISE_FULL)),
)
OLD_STAGE_MIX = 0.2  # share of training episodes taken from easier stages, so skills are not forgotten
FIRST_CLEAR = (-45, Y - 16, -45, 45, Y + 16, 45)  # the first round's arena check covers this much
LEFTOVERS = [-110, Y - 60, -110, 150, Y + 90, 150]  # all an earlier run in this game can have built, also the map
FIRST_JUMP_BONUS = 1.0  # once per episode on a jump stage, so it is worth trying and cannot be farmed
EMPTY_CLICK_COST = 0.05  # per click with no blocks left, which can never do anything
STAYED_HOME_COST = 2.0  # time ran out and the agent never left the start platform behind
AIM_REWARD = 0.5  # on jump stages, for turning toward the goal. Turning away gives it back, so it cannot be farmed.
STALL_COST = 1.0
GOAL_REWARD = 10.0
SPEED_BONUS = 10.0
TICK_COST = 0.006  # per tick: 10 blocks walked at half speed instead of sprinting costs about as much as a wasted block
AIMED_AT_GOAL = 0.87  # within about 30 degrees of looking at the goal
JUMP_SHIFT_COST = 0.05  # per tick with shift on a jump stage, where there is nothing to place
MAX_DEATHS = 2  # the round ends on this death, without a respawn
RESPAWN_TICKS = 80  # 4 seconds of game time spent dead
CLIMB_GAP = (0, 1)  # empty blocks between steps when going up
HOP_GAP = (1, 1)  # empty blocks between single blocks on the same level: this is what a test could clear reliably
POLE_HEIGHT = 4
LANDING_BONUS = 1.0  # per block landed on, only in order: skipping one ends these points for that round
WIDE_ZIGZAG_GAP = (1, 1)  # one empty block ahead and two sideways: a truly diagonal jump with a real hole under it
ISLAND_GAP = (1, 1)  # empty blocks between the single blocks it has to stop on
STOP_BONUS = 0.3  # the first time it stands still on each island block
STALL_TICKS = 40  # 2 seconds of game time
# The deepest a goal may lie below the start: a straight drop of 12 costs 9 of 20 health. Blocks cannot be built
# downward into the air, so a goal further down is reached only by a fall that kills (24 down: 21 damage). Stairs
# going down drew such goals in 29 % of rounds at the top level, and treenan now and then (down to 19).
MAX_DROP = 12
DAMAGE_SETTLE_TICKS = 3  # the server reports landing damage a moment after the client lands
AXIS, EIGHT, ANY = "axis", "eight", "any"
KEYS = ("forward", "back", "left", "right", "sneak", "jump", "sprint")

EDGE, CENTER = "edge", "center"

ISLAND, PILLAR, WALL, SLALOM, MIXED, JUMP = "island", "pillar", "wall", "slalom", "mixed", "jump"
RUN, RUN_HOLES = "run", "run_holes"  # a narrow winding path to the goal, then the same with holes to jump
TREENAN = "treenan"  # from beside the bed to a grass spot on another island of the real map
CLIMB, HOPS, ISLANDS, WIDE_ZIGZAG = "climb", "hops", "islands", "wide_zigzag"  # up, in a row, to stop on, offset
COURSES = (CLIMB, HOPS, ISLANDS, WIDE_ZIGZAG)  # made of single blocks to jump between
STOP_COURSES = (ISLANDS, WIDE_ZIGZAG)  # the ones that pay for stopping on each block
COURSE_GAP = {CLIMB: CLIMB_GAP, HOPS: HOP_GAP, ISLANDS: ISLAND_GAP, WIDE_ZIGZAG: WIDE_ZIGZAG_GAP}
NO_BLOCKS = (JUMP, RUN, RUN_HOLES, *COURSES)  # stages played without blocks, and without shift
OBSTACLE_KINDS = (None, ISLAND, PILLAR, WALL, SLALOM)

# Blocks handed out per task: (times the estimated need, plus this many). Tighter further on.
ROOMY, NORMAL, TIGHT, STRICT, NONE = (3.0, 2), (2.0, 2), (1.75, 2), (1.5, 2), (0.0, 0)

# Phases of a skill: learn it (dying is cheap, helper points on), make it safe (dying costs more, helpers
# gone), make it efficient (dying costs most, blocks and time count too).
LEARN, SECURE, EFFICIENT = "learn", "secure", "efficient"
DEATH_COST = {LEARN: 2.0, SECURE: 5.0, EFFICIENT: 8.0}

# (gap, directions, height changes, yaw error at start, ticks per block, start, extras, block budget, phase)
STAGES = [
    (1, AXIS, FLAT, 15, 40, EDGE, None, ROOMY, LEARN),
    (1, AXIS, FLAT, 60, 40, EDGE, None, ROOMY, LEARN),
    (2, AXIS, FLAT, 180, 40, EDGE, None, ROOMY, LEARN),
    (2, AXIS, FLAT, 180, 40, CENTER, None, ROOMY, SECURE),
    (1, AXIS, FLAT, 180, 40, CENTER, JUMP, NONE, LEARN),
    (2, AXIS, FLAT, 15, 40, CENTER, JUMP, NONE, LEARN),  # facing the goal, with room for a run-up
    (2, AXIS, FLAT, 45, 40, CENTER, JUMP, NONE, LEARN),
    (2, AXIS, FLAT, 90, 40, CENTER, JUMP, NONE, SECURE),
    (2, AXIS, FLAT, 180, 40, CENTER, JUMP, NONE, SECURE),
    # Back to building. Gap 5 cannot be jumped in any direction (measured), gap 3 can only when straight.
    (5, AXIS, FLAT, 15, 40, EDGE, None, ROOMY, LEARN),
    (5, AXIS, FLAT, 15, 40, CENTER, None, ROOMY, LEARN),
    (5, AXIS, FLAT, 90, 40, CENTER, None, ROOMY, LEARN),
    (5, AXIS, FLAT, 180, 40, CENTER, None, ROOMY, SECURE),
    (5, EIGHT, FLAT, 180, 40, CENTER, None, NORMAL, LEARN),
    (5, ANY, FLAT, 180, 40, CENTER, None, NORMAL, SECURE),
    (14, AXIS, FLAT, 180, 20, CENTER, RUN, NONE, LEARN),
    (14, AXIS, FLAT, 180, 20, CENTER, RUN_HOLES, NONE, LEARN),
    (8, AXIS, FLAT, 180, 30, CENTER, CLIMB, NONE, LEARN),
    (4, AXIS, FLAT, 180, 40, CENTER, WIDE_ZIGZAG, NONE, LEARN),  # real holes: 2 ahead and 2 sideways, too far apart to touch both rows on a straight run
    (8, AXIS, FLAT, 180, 30, CENTER, HOPS, NONE, LEARN),
    (8, AXIS, FLAT, 180, 30, CENTER, ISLANDS, NONE, LEARN),
    (8, ANY, FLAT, 180, 40, CENTER, None, NORMAL, SECURE),
    (8, ANY, (-1, 0, 1), 180, 40, CENTER, None, NORMAL, SECURE),
    (8, ANY, HILLS, 180, 40, CENTER, None, TIGHT, SECURE),
    (8, ANY, DROPS, 180, 40, CENTER, None, TIGHT, SECURE),
    (8, ANY, FLAT, 180, 40, CENTER, ISLAND, TIGHT, LEARN),
    (8, ANY, FLAT, 180, 40, CENTER, PILLAR, TIGHT, LEARN),
    (8, ANY, FLAT, 180, 40, CENTER, WALL, TIGHT, LEARN),
    (8, ANY, FLAT, 180, 40, CENTER, SLALOM, TIGHT, LEARN),
    (8, ANY, HILLS, 180, 40, CENTER, MIXED, TIGHT, SECURE),
    (20, ANY, DROPS, 180, 40, CENTER, TREENAN, TIGHT, LEARN),
    (20, ANY, DROPS, 180, 40, CENTER, TREENAN, TIGHT, SECURE),
    (12, ANY, DROPS, 180, 30, CENTER, MIXED, TIGHT, EFFICIENT),
    (12, ANY, DROPS, 180, 20, CENTER, MIXED, STRICT, EFFICIENT),
    (12, ANY, DROPS, 180, 15, CENTER, MIXED, STRICT, EFFICIENT),
    (12, ANY, DROPS, 180, 10, CENTER, MIXED, STRICT, EFFICIENT),
    (12, ANY, DROPS, 180, 7, CENTER, MIXED, STRICT, EFFICIENT),
]
LAST_JUMP_STAGE = max(i for i, t in enumerate(STAGES) if t[6] == JUMP)
BLOCK_GOAL_FROM_STAGE = next(i for i, t in enumerate(STAGES) if t[8] == EFFICIENT)
HEAD = 35  # values before the block cube


def pace(next_time, seconds):
    """Keeps a steady speed for people watching. Time lost, such as being dead, is never caught up."""
    next_time = max(next_time + seconds, time.perf_counter())
    time.sleep(max(0.0, next_time - time.perf_counter()))
    return next_time


def rotate(x, z, k):
    """Turns a horizontal vector so that yaw decreases by 90 * k degrees."""
    for _ in range(k % 4):
        x, z = z, -x
    return x, z


def _grid_index(k):
    size = 2 * R + 1
    index = np.zeros(size**3, dtype=np.int64)
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            for z in range(-R, R + 1):
                nx, nz = rotate(x, z, k)
                new = (nx + R) * size * size + (y + R) * size + (nz + R)
                index[new] = (x + R) * size * size + (y + R) * size + (z + R)
    return index


GRID = [_grid_index(k) for k in range(4)]


class SkillEnv(gym.Env):
    landing_order = ()
    course = ()
    course_axis = (1, 0)
    stop_cells = frozenset()
    map_data = None
    map_cells = frozenset()
    next_cells = {}
    goal_half, field_margin, fall_margin, walk_par = 0, 6, 1.5, 0
    goal_size = "platform"  # a win is standing on the goal block itself, not just near it
    ticks_per_decision = TICKS_PER_DECISION
    max_sneak_ticks = MAX_SNEAK_TICKS
    turn_choices = len(YAW_STEPS)
    realtime = False  # True makes a respawn take 4 real seconds, for people watching
    map_gaps = (MIN_GAP, MAX_GAP)  # blocks between the islands on the real map
    forced_task = None  # plays this stage's task regardless of the training stage, for skill tests
    reward_scale = 0.1  # keeps returns near the range the value network starts in
    camera_every = 0.0  # seconds between viewer moves. Fast training raises this so the view does not jerk.
    N_OBS = HEAD + (2 * R + 1) ** 3 + 8 * HISTORY

    def __init__(self, port, stage=0, render=False, mix=0.0):
        self.c = McClient(port)
        self.c.render(render)
        self.c.call(op="wipe", box=LEFTOVERS)  # this process has no record of what the last one built
        self.mix = mix
        self.dirty = list(FIRST_CLEAR)
        self.set_stage(stage)
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (self.N_OBS,), np.float32)
        # forward/back, strafe, sneak, clicks this decision (0-2), jump, sprint, yaw, pitch
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2, 3, 2, 2, len(YAW_STEPS), len(PITCH_STEPS)])

    def set_stage(self, stage):
        self.stage = stage
        self._use_task(stage)
        for first, level in HUMAN_LEVELS:
            if stage >= first:
                self.delay_low, self.delay_high, self.late_chance, self.aim_noise = level

    def set_mix(self, mix):
        self.mix = mix

    def sidebar(self, title, lines):
        self.c.call(op="sidebar", title=title, lines=lines)

    def set_forced_task(self, stage):
        self.forced_task = stage

    def _pick_stage(self):
        """Mostly the current stage, sometimes an easier one. Building and jumping pull in opposite
        directions, so old jump rounds are never mixed into building stages, or the other way around."""
        if self.forced_task is not None:
            return self.forced_task
        if self.np_random.random() >= self.mix:
            return self.stage
        no_blocks = STAGES[self.stage][6] in NO_BLOCKS
        older = [i for i in range(self.stage) if (STAGES[i][6] in NO_BLOCKS) == no_blocks]
        return int(self.np_random.choice(older)) if older else self.stage

    def _use_task(self, stage):
        self.played_stage = stage
        if STAGES[stage][6] == TREENAN:
            self.goal_half, self.field_margin, self.fall_margin = 0, 18, 6.0
        self.kind, self.blocked, self.detour, self.field = None, set(), 0, None
        self.gap, self.directions, self.heights, self.yaw_range, self.ticks_per_block, self.start, self.extras, self.budget, _ = STAGES[stage]

    def _direction(self):
        r = self.np_random
        if self.directions == AXIS:
            angle = r.integers(4) * math.pi / 2
        elif self.directions == EIGHT:
            angle = r.integers(8) * math.pi / 4
        else:
            angle = r.uniform(0, 2 * math.pi)
        return math.cos(angle), math.sin(angle)

    def _treenan_spec(self):
        """One round on the real map. Blocks that would be in the way of walking steer the route around them."""
        r = self.np_random
        if getattr(self, "map_kind", "treenan") == "bedwars":
            return self._bedwars_spec()
        if SkillEnv.map_data is None:
            SkillEnv.map_data = Treenan()
        data = SkillEnv.map_data
        for _ in range(40):
            target = "straight" if r.random() < 0.5 else "diagonal"
            gap = int(r.integers(self.map_gaps[0], self.map_gaps[1] + 1))
            most = getattr(self, "map_rise", 0)
            rise = int(r.integers(-most, most + 1)) if most else 0  # the other island moved up or down
            start = data.homes[int(r.integers(len(data.homes)))] if getattr(self, "map_start", "bed") == "anywhere" else None
            any_top = getattr(self, "map_spots", "grass") == "any top"
            placed, (gx, gy, gz) = data.layout(target, gap, r, Y, rise, start, any_top)
            # Climbing costs blocks beyond the width of the gap. One stack of 64 is all it holds, so a round that
            # would need more gets a shorter gap instead of too few blocks.
            fits = int((64 - self.budget[1] - getattr(self, "push_spare", 0)) / max(self.budget[0], 1e-9)) - 2 * max(gy - Y, 0) - self.descent_blocks(gy - Y)
            if self.budget[0] and gap > max(fits, MIN_GAP):
                gap = max(fits, MIN_GAP)
                placed, (gx, gy, gz) = data.layout(target, gap, r, Y, rise, start, any_top)
            self.goal_x, self.goal_z, self.dy = gx, gz, gy - Y
            if self.dy < -MAX_DROP:  # reached only by a deadly fall (auto_env.MAX_DROP): another spot
                continue
            other = data.last_other
            self.blocked = {(x, z) for (x, y, z) in placed if ((gy + 2 <= y <= gy + 3) if (x, y, z) in other else (Y + 2 <= y <= Y + 3))}
            self.blocked -= {(x, z) for x in range(-1, 2) for z in range(-1, 2)}
            self.blocked -= {(x, z) for x in range(gx - 1, gx + 2) for z in range(gz - 1, gz + 2)}
            self._build_field()
            if self.field is None or self._next_cell(0.5, 0.5)[1] is not None and self._walk(0.5, 0.5) < self.field.max() - 1.0:  # nothing in the way: fine
                break
        else:
            raise RuntimeError("no reachable grass spot found")
        self.map_target, self.map_gap = target, gap
        # Up costs about two blocks a step: a pillar to stand on and the way on from it.
        self.par = gap + 2 * max(self.dy, 0) + self.descent_blocks(self.dy)
        self.walk_par = int(self._walk(0.5, 0.5))
        self.blocks_given = min(64, math.ceil(self.par * self.budget[0]) + self.budget[1])
        self.next_cells = placed
        yaw = float(r.uniform(0, 360))
        return dict(clear=[0, 0, 0, -1, -1, -1], platforms=[], skip_block_check=True, camera=self.camera_for(gx, gz, gy), camera_every=self.camera_every, beam=[gx + 0.5, gy + 1.2, gz + 0.5],
                    pos=[0.5 + float(r.uniform(-0.3, 0.3)), Y + 1, 0.5 + float(r.uniform(-0.3, 0.3))], yaw=yaw, pitch=float(r.uniform(-30, 89)),
                    blocks=self.blocks_given, block_radius=R)

    def _prepare_arena(self):
        """The map is pasted into the world before the round, and everything from the last round is removed."""
        is_map = self.extras == TREENAN
        if not is_map and not self.map_cells:
            return
        air = dict.fromkeys(self.map_cells, "minecraft:air")
        new = self.next_cells if is_map else {}
        if is_map:
            self.c.call(op="wipe", box=self.dirty)
            self.dirty = [-1, Y, -1, 1, Y, 1]
        rows, palette, index = [], [], {}
        for (x, y, z), state in {**air, **new}.items():
            if state not in index:
                index[state] = len(palette)
                palette.append(state)
            rows.append([x, y, z, index[state]])
        self.c.call(op="paste", palette=palette, blocks=rows)
        self.map_cells = set(new)

    def _spec(self):
        self.stop_cells = set()
        self.landing_order = []
        if self.extras == TREENAN:
            return self._treenan_spec()
        self.goal_half, self.field_margin, self.fall_margin, self.walk_par = 0, 6, 1.5, 0
        r = self.np_random
        ux, uz = self._direction()
        # The blocks have to cover this way at its highest step up, around an obstacle, plus any spare for pushes,
        # inside one stack of 64. A diagonal that would need more gets shorter; a straight one keeps its length.
        building = self.extras not in NO_BLOCKS
        detour = 4 if building and self.extras is not None else 0
        reach = self.gap + 4
        while True:
            scale = reach / max(abs(ux), abs(uz))
            self.goal_x = int(round(ux * scale))
            self.goal_z = int(round(uz * scale))
            worst = max(self.blocks_needed(self.goal_x, self.goal_z, max(max(self.heights), 0)),
                        self.blocks_needed(self.goal_x, self.goal_z, 0) + self.descent_blocks(min(self.heights))) + detour + self._single_extra(self.goal_x, self.goal_z)
            if not building or not self.budget[0] or reach <= 9 or math.ceil(worst * self.budget[0]) + self.budget[1] + getattr(self, "push_spare", 0) <= 64:
                break
            reach -= 1
        self.dy = int(r.choice(self.heights))
        path = None
        course = None
        if self.extras in (RUN, RUN_HOLES):
            path, holes, self.goal_x, self.goal_z = self._make_path(ux, uz, self.extras == RUN_HOLES)
        elif self.extras in COURSES:
            course, holes, self.goal_x, self.goal_z, self.dy = self._make_course(ux, uz, self.extras)
            path = [(x, z) for x, _, z in course]
            self.course, self.course_axis = course, (int(round(ux)), int(round(uz)))
            self.landing_order = [(x, y + 1, z) for x, y, z in course]
            lowest = min(min(y for _, y, _ in course) + 1, Y + 1)
            self.fall_margin = 1.5 + min(Y + 1, Y + 1 + self.dy) - lowest
            self.stop_cells = {(x, y + 1, z) for x, y, z in course} if self.extras in STOP_COURSES else set()
        gx, gz, gy = self.goal_x, self.goal_z, Y + self.dy
        small = self.yaw_range <= 15
        faces_goal = self.extras in NO_BLOCKS  # the error is measured from looking at the goal on these stages
        yaw = math.degrees(math.atan2(-ux, uz)) + (0.0 if faces_goal else 180.0) + r.uniform(-self.yaw_range, self.yaw_range)
        if faces_goal:
            pitch = float(r.uniform(-5, 15)) if small else float(r.uniform(-30, 60))
        else:
            pitch = 80.0 + r.uniform(-5, 5) if small else float(np.clip(80.0 + r.uniform(-70, 70), -30, 89))
        kind = None if self.extras in NO_BLOCKS else self.extras
        if kind == MIXED:
            kind = OBSTACLE_KINDS[int(r.integers(len(OBSTACLE_KINDS)))]
        self.kind = kind
        extra_boxes, self.blocked, self.detour = self._obstacles(kind, ux, uz, gx, gz, gy)
        if path is not None:
            extra_boxes = [[x, y, z, x, y, z] for x, y, z in course] if course else [[x, Y, z, x, Y, z] for x, z in path]
            if self.extras == WIDE_ZIGZAG:
                extra_boxes += [[x, Y - 1, z, x, Y + POLE_HEIGHT, z] for x, z in self.poles]
            self.blocked = self._off_path(path, holes)
        self._build_field()
        self.par = len(path) if path is not None else max(1, self.blocks_needed(gx, gz, self.dy) + self.descent_blocks(self.dy) + self.detour + self._single_extra(gx, gz))
        self.blocks_given = min(64, math.ceil(self.par * self.budget[0]) + self.budget[1])
        m = 8
        box = [min(-1, gx - 2) - m, min(Y, gy) - m, min(-1, gz - 2) - m, max(1, gx + 2) + m, max(Y, gy) + m, max(1, gz + 2) + m]
        # The last arena can lie in another direction, so it has to be cleared along with the new one.
        clear = [min(a_, b_) for a_, b_ in zip(self.dirty[:3], box[:3])] + [max(a_, b_) for a_, b_ in zip(self.dirty[3:], box[3:])]
        self.dirty = box
        px, pz = 0.5 + float(r.uniform(-0.3, 0.3)), 0.5 + float(r.uniform(-0.3, 0.3))
        if self.start == EDGE:
            # The platform edge is 1.5 from its centre. Start 0.2 to 0.5 from it, facing the other way.
            along = float(r.uniform(1.0, 1.3))
            px, pz = 0.5 + ux * along + float(r.uniform(-0.1, 0.1)) * uz, 0.5 + uz * along + float(r.uniform(-0.1, 0.1)) * ux
        return dict(
            clear=clear,
            camera=self.camera_for(gx, gz, gy),
            camera_every=self.camera_every,
            beam=[gx + 0.5, gy + 1.2, gz + 0.5],
            platforms=[[-1, Y, -1, 1, Y, 1], [gx, gy, gz, gx, gy, gz] if self.goal_size == "single block" else [gx - 2, gy, gz - 2, gx + 2, gy, gz + 2], *extra_boxes],
            pos=[px, Y + 1, pz],
            yaw=float(yaw % 360), pitch=pitch, blocks=self.blocks_given, block_radius=R,
        )

    @staticmethod
    def camera_for(gx, gz, gy):
        """A spot to the side of the route and above it, looking at the middle with both platforms in view."""
        mx, my, mz = gx / 2 + 0.5, (Y + 1 + gy + 1) / 2, gz / 2 + 0.5
        length = max(math.hypot(gx, gz), 1.0)
        side_x, side_z = gz / length, -gx / length
        # Fits the route plus both platforms on screen at FOV 90, without standing further back than needed.
        away = max(7.0, length * 0.5 + 2.5)
        up = away * 0.4
        x, z, y = mx + side_x * away, mz + side_z * away, my + up
        yaw = math.degrees(math.atan2(-(mx - x), mz - z))
        pitch = math.degrees(math.atan2(up, away))
        return [x, y, z, yaw, pitch]

    def _make_path(self, ux, uz, with_holes):
        """A one block wide path that winds sideways on its way to the goal. Holes sit only on straight parts,
        with room to run up and land, and are 1 or 2 blocks long."""
        r = self.np_random
        main_x, main_z = int(round(ux)), int(round(uz))
        side_x, side_z = main_z, main_x
        along, lat, path, holes = 2, 0, [], []
        straight, next_turn = 0, int(r.integers(2, 5))
        cell = lambda a, s: (main_x * a + side_x * s, main_z * a + side_z * s)
        end = 2 + self.gap
        while along < end:
            path.append(cell(along, lat))
            straight += 1
            room_left = end - along
            if with_holes and straight >= 3 and room_left >= 5 and r.random() < 0.35:
                size = int(r.integers(1, 3))
                for i in range(size):
                    holes.append(cell(along + 1 + i, lat))
                along += size + 1
                path.append(cell(along, lat))  # landing
                along += 1
                path.append(cell(along, lat))
                straight, next_turn = 1, max(next_turn, 2)
                along += 1
                continue
            if straight >= next_turn and room_left >= 3:
                step = int(r.choice((-1, 1)))
                if abs(lat + step) > 4:
                    step = -step
                lat += step
                path.append(cell(along, lat))
                straight, next_turn = 0, int(r.integers(2, 5))
            along += 1
        return path, holes, *cell(along + 2, lat)

    def _make_course(self, ux, uz, kind):
        """Single blocks to jump between. Distances are limited to what a test in the real game could clear."""
        r = self.np_random
        main_x, main_z = int(round(ux)), int(round(uz))
        side_x, side_z = main_z, main_x
        cell = lambda a, s: (main_x * a + side_x * s, main_z * a + side_z * s)
        along, lat, height = 1, 0, Y
        course, holes = [], []
        last_change = 0
        self.poles = []
        for _ in range(self.gap):
            low, high = COURSE_GAP[kind]
            gap = int(r.integers(low, high + 1))
            for i in range(gap):
                holes.append(cell(along + 1 + i, lat))
            along += gap + 1
            if kind == WIDE_ZIGZAG:
                # The next block sits two to the side, back and forth, so going straight on misses it.
                new_lat = 2 - lat
                for side in range(min(lat, new_lat), max(lat, new_lat) + 1):
                    holes.append(cell(along, side))
                # A pole in the row it just left, so it cannot skip along one row instead of zigzagging.
                self.poles.append(cell(along, lat))
                lat = new_lat
            elif kind == CLIMB:
                height += 1
            elif kind == ISLANDS:
                # Small random changes in height, never two steps the same way in a row, so it is no staircase.
                change = int(r.choice((-1, 0, 0, 1)))
                if change == last_change and change != 0:
                    change = 0
                if abs(height + change - Y) > 2:
                    change = 0
                height += change
                last_change = change
            x, z = cell(along, lat)
            course.append((x, height, z))
        # A landing pad, with one gap's distance to it.
        last_gap = 1
        along += 1 + last_gap
        for i in range(along - last_gap, along):
            holes.append(cell(i, lat))
        goal_x, goal_z = cell(along + 2, lat)
        return course, holes, goal_x, goal_z, height - Y

    def _off_path(self, path, holes):
        """Every column the way to the goal may not pass, so the route follows the path."""
        allowed = set(path) | set(holes)
        allowed |= {(x, z) for x in range(-1, 2) for z in range(-1, 2)}
        allowed |= {(x, z) for x in range(self.goal_x - 2, self.goal_x + 3) for z in range(self.goal_z - 2, self.goal_z + 3)}
        m = 6
        return {(x, z)
                for x in range(min(-1, self.goal_x - 2) - m, max(1, self.goal_x + 2) + m + 1)
                for z in range(min(-1, self.goal_z - 2) - m, max(1, self.goal_z + 2) + m + 1)
                if (x, z) not in allowed}

    def _obstacles(self, kind, ux, uz, gx, gz, gy):
        """Stone placed along the straight line to the goal. Returns boxes, blocked columns, block change."""
        if kind is None:
            return [], set(), 0
        boxes, blocked = [], set()
        low, high = min(Y, gy) - 1, max(Y, gy) + 4
        along_x = abs(gx) >= abs(gz)
        flip = int(self.np_random.choice((-1, 1)))

        def point(t, offset=0):
            cx, cz = int(round(gx * t)), int(round(gz * t))
            return (cx, cz + offset) if along_x else (cx + offset, cz)

        def wall(t, half, offset=0):
            cx, cz = point(t, offset)
            x0, x1, z0, z1 = (cx, cx, cz - half, cz + half) if along_x else (cx - half, cx + half, cz, cz)
            boxes.append([x0, low, z0, x1, high, z1])
            blocked.update((x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1))

        if kind == ISLAND:
            cx, cz = point(0.5)
            height = int(round((Y + gy) / 2))
            boxes.append([cx - 1, height, cz - 1, cx + 1, height, cz + 1])
            return boxes, blocked, -3  # three fewer blocks when the route crosses it
        if kind == PILLAR:
            wall(0.5, 0)
            return boxes, blocked, 2
        if kind == WALL:
            wall(0.5, 2)
            return boxes, blocked, 4
        wall(0.35, 1, flip)  # slalom
        wall(0.65, 1, -flip)
        return boxes, blocked, 4

    def _build_field(self):
        """Walking distance to the goal around obstacles. Without obstacles it is the straight line."""
        self.field = None
        if not self.blocked:
            return
        m = self.field_margin
        self.fx0, self.fz0 = min(-1, self.goal_x - 2) - m, min(-1, self.goal_z - 2) - m
        nx = max(1, self.goal_x + 2) + m - self.fx0 + 1
        nz = max(1, self.goal_z + 2) + m - self.fz0 + 1
        dist = np.full((nx, nz), np.inf)
        queue = []
        h = self.goal_half
        for x in range(self.goal_x - h, self.goal_x + h + 1):
            for z in range(self.goal_z - h, self.goal_z + h + 1):
                dist[x - self.fx0, z - self.fz0] = 0.0
                queue.append((0.0, x - self.fx0, z - self.fz0))
        heapq.heapify(queue)
        blocked = {(x - self.fx0, z - self.fz0) for x, z in self.blocked}
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        while queue:
            d, i, j = heapq.heappop(queue)
            if d > dist[i, j]:
                continue
            for di, dj, cost in steps:
                a, b = i + di, j + dj
                if not (0 <= a < nx and 0 <= b < nz) or (a, b) in blocked:
                    continue
                # No cutting corners of an obstacle.
                if di and dj and ((a, j) in blocked or (i, b) in blocked):
                    continue
                if d + cost < dist[a, b]:
                    dist[a, b] = d + cost
                    heapq.heappush(queue, (d + cost, a, b))
        far = dist[np.isfinite(dist)].max() + 2.0
        self.field = np.where(np.isfinite(dist), dist, far)

    def _next_cell(self, x, z):
        """The neighbouring column closest to the goal on foot, and the walking distance through it."""
        nx, nz = self.field.shape
        ci, cj = int(math.floor(x)) - self.fx0, int(math.floor(z)) - self.fz0
        best, best_cell = float("inf"), None
        for i in range(max(0, ci - 1), min(nx, ci + 2)):
            for j in range(max(0, cj - 1), min(nz, cj + 2)):
                if (i + self.fx0, j + self.fz0) in self.blocked:
                    continue
                # Corners of blocked columns cannot be cut.
                if i != ci and j != cj and ((i + self.fx0, cj + self.fz0) in self.blocked or (ci + self.fx0, j + self.fz0) in self.blocked):
                    continue
                cx, cz = i + self.fx0 + 0.5, j + self.fz0 + 0.5
                total = self.field[i, j] + math.hypot(cx - x, cz - z)
                if total < best:
                    best, best_cell = total, (cx, cz)
        return best, best_cell

    def _walk(self, x, z):
        """Distance left on the ground."""
        if self.field is None:
            return math.hypot(self.goal_x + 0.5 - x, self.goal_z + 0.5 - z)
        best, _ = self._next_cell(x, z)
        return best if best < float("inf") else float(self.field.max())

    def _next_block(self, x, z):
        """On single block courses the way to go is the next block itself, not the air in between."""
        along = x * self.course_axis[0] + z * self.course_axis[1]
        for cx, _, cz in self.course:
            if (cx + 0.5) * self.course_axis[0] + (cz + 0.5) * self.course_axis[1] > along + 0.4:
                return cx + 0.5, cz + 0.5
        return self.goal_x + 0.5, self.goal_z + 0.5

    def _route(self, x, z):
        """Unit vector along the way to go: straight at the goal, or toward the next column of the way around."""
        if self.extras in COURSES and self.course:
            tx, tz = self._next_block(x, z)
            norm = math.hypot(tx - x, tz - z) or 1.0
            return (tx - x) / norm, (tz - z) / norm
        if self.field is None:
            dx, dz = self.goal_x + 0.5 - x, self.goal_z + 0.5 - z
        else:
            _, cell = self._next_cell(x, z)
            if cell is None:
                dx, dz = self.goal_x + 0.5 - x, self.goal_z + 0.5 - z
            else:
                # Inside the column it is headed for, aim at the next one on from there.
                if math.hypot(cell[0] - x, cell[1] - z) < 0.35:
                    _, further = self._next_cell(cell[0], cell[1])
                    i, j = int(math.floor(cell[0])) - self.fx0, int(math.floor(cell[1])) - self.fz0
                    best = self.field[i, j]
                    if further is not None and self.field[int(math.floor(further[0])) - self.fx0, int(math.floor(further[1])) - self.fz0] < best:
                        cell = further
                    else:
                        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                            a, b = i + di, j + dj
                            if 0 <= a < self.field.shape[0] and 0 <= b < self.field.shape[1] and (a + self.fx0, b + self.fz0) not in self.blocked and self.field[a, b] < best:
                                best = self.field[a, b]
                                cell = (a + self.fx0 + 0.5, b + self.fz0 + 0.5)
                dx, dz = cell[0] - x, cell[1] - z
        norm = math.hypot(dx, dz) or 1.0
        return dx / norm, dz / norm

    def _single_extra(self, gx, gz):
        """A lone goal block covers 2 less on each side than the 5x5 platform blocks_needed counts on."""
        return (2 * (gx != 0) + 2 * (gz != 0)) if self.goal_size == "single block" else 0

    @staticmethod
    def descent_blocks(dy):
        """Blocks for a way down with no drop over 3, which costs no health: one landing every 3 blocks."""
        return math.ceil((-dy - 3) / 3) if dy < -3 else 0

    @staticmethod
    def blocks_needed(goal_x, goal_z, dy):
        """Estimate of the blocks for a walkable bridge on the grid. Jumping can legitimately beat it."""
        # Platforms cover -1..1 and goal-2..goal+2, so 4 of the distance is already solid ground.
        sx, sz = max(abs(goal_x) - 4, 0), max(abs(goal_z) - 4, 0)
        across = sx + sz + (1 if sx and sz else 0)  # a walkable bridge that goes both ways turns a corner: one more
        return max(1, across + max(dy, 0))

    def _touched(self, o):
        """Marks placed blocks that touch the player's hitbox. Generous, so it never overcounts waste."""
        x, y, z = o["pos"]
        for b in self.placed:
            if (b[0] - 0.01 < x + 0.3 and x - 0.3 < b[0] + 1.01 and b[1] - 0.01 < y + 1.8 and y < b[1] + 1.01
                    and b[2] - 0.01 < z + 0.3 and z - 0.3 < b[2] + 1.01):
                self.touched.add(b)

    def unused_blocks(self):
        """Placed blocks the player never touched, and that no touched block was built on."""
        useful = set(self.touched)
        stack = list(useful)
        while stack:
            support = self.placed.get(stack.pop())
            if support in self.placed and support not in useful:
                useful.add(support)
                stack.append(support)
        return len(self.placed) - len(useful)

    def _distance(self, o):
        x, y, z = o["pos"]
        return self._walk(x, z) + abs(Y + 1 + self.dy - y)

    def reached(self, o):
        x, y, z = o["pos"]
        reach = self.goal_half + 0.5
        return (o["on_ground"] and abs(x - self.goal_x - 0.5) <= reach and abs(z - self.goal_z - 0.5) <= reach
                and abs(y - (Y + 1 + self.dy)) < 0.1)

    def fell(self, o):
        return o["pos"][1] < min(Y + 1, Y + 1 + self.dy) - self.fall_margin

    @staticmethod
    def _on_start_platform(o):
        x, y, z = o["pos"]
        return max(abs(x - 0.5), abs(z - 0.5)) <= 1.8 and abs(y - (Y + 1)) < 0.6

    def _way_in_view(self, o):
        """Where to go, seen from the eyes: ahead is 1 when looking straight along it, side is its offset."""
        x, _, z = o["pos"]
        return self._in_view(*self._route(x, z), o["yaw"])

    @staticmethod
    def _in_view(rx, rz, yaw_degrees):
        yaw = math.radians(yaw_degrees)
        return rx * -math.sin(yaw) + rz * math.cos(yaw), rx * -math.cos(yaw) + rz * -math.sin(yaw)

    def _at_edge_toward_goal(self, o):
        """Within 0.9 of the start platform's edge and moving toward the goal. Only such a jump can clear a gap."""
        x, _, z = o["pos"]
        near_edge = max(abs(x - 0.5), abs(z - 0.5)) >= 0.6
        toward = o["vel"][0] * (self.goal_x + 0.5 - x) + o["vel"][2] * (self.goal_z + 0.5 - z)
        return near_edge and toward > 0 and self._way_in_view(o)[0] >= AIMED_AT_GOAL

    @staticmethod
    def _is_on(o, cell):
        """Standing on this block, also when partly over its edge."""
        x, y, z = o["pos"]
        return (int(math.floor(y)) == cell[1] and cell[0] - 0.3 < x < cell[0] + 1.3 and cell[2] - 0.3 < z < cell[2] + 1.3)

    def _standing_on(self, o):
        """The island block under the player, also when it stands partly over the edge of it."""
        return next((cell for cell in self.stop_cells if self._is_on(o, cell)), None)

    def _with_timer(self, seen):
        """A player always knows how long they have held a key, so this one value is not delayed.
        0 is shift up. Held, it falls from 1 to 0.2 as the second runs out."""
        if self.keys["sneak"]:
            seen[SNEAK_SLOT] = 1.0 - 0.8 * self.sneak_held / MAX_SNEAK_TICKS
        else:
            seen[SNEAK_SLOT] = 0.0
        seen[KNOWN_SLOT] = max(0.0, 1.0 - self.t / self.limit)  # time left of the round
        seen[KNOWN_SLOT + 1] = max(0.0, 1.0 - self.since_best / STALL_TICKS)  # time left before it counts as stuck
        seen[KNOWN_SLOT + 2] = self.deaths / MAX_DEATHS
        seen[KNOWN_SLOT + 3] = float(self.keys["sprint"])
        seen[KNOWN_SLOT + 4] = float(self.keys["jump"])
        return seen

    def _edge_left(self, o):
        """How far along the way to go the ground under the feet continues, up to 2 blocks. Seeing where the
        edge is matters most for tapping shift, and is hard to work out from the raw block cube."""
        solid = o.get("solid")
        feet = o.get("feet")
        if not solid or not feet:
            return 1.0
        x, _, z = o["pos"]
        rx, rz = self._route(x, z)
        size = 2 * R + 1
        for i in range(0, 21):
            d = i * 0.1
            cx = math.floor(x + rx * d) - feet[0]
            cz = math.floor(z + rz * d) - feet[2]
            if abs(cx) > R or abs(cz) > R:
                return 1.0
            if solid[(int(cx) + R) * size * size + (R - 1) * size + (int(cz) + R)] != "1":
                return d / 2
        return 1.0

    def _place_gain(self, o):
        """Would a click put a block closer to the goal than the one it is placed on? From -1 to 1."""
        h = o.get("hit") or {}
        face = FACES.get(h.get("face"))
        if h.get("type") != "block" or face is None:
            return 0.0
        b = h["block"]
        before = math.hypot(self.goal_x - b[0], self.goal_z - b[2])
        after = math.hypot(self.goal_x - b[0] - face[0], self.goal_z - b[2] - face[1])
        return before - after

    def _look_ahead(self, o):
        """The way to go 3 blocks further on, seen from the eyes, so turns around obstacles come early."""
        x, _, z = o["pos"]
        for _ in range(6):
            rx, rz = self._route(x, z)
            x, z = x + rx * 0.5, z + rz * 0.5
        return self._in_view(*self._route(x, z), o["yaw"])

    def _vec(self, o):
        x, y, z = o["pos"]
        vx, vy, vz = o["vel"]
        k = int(round(o["yaw"] / 90.0))
        residual = math.radians(o["yaw"] - k * 90.0)
        gx, gz = rotate(self.goal_x + 0.5 - x, self.goal_z + 0.5 - z, k)
        fx, fz = rotate(x - math.floor(x) - 0.5, z - math.floor(z) - 0.5, k)
        ahead, side = self._way_in_view(o)
        cvx, cvz = rotate(vx, vz, k)
        h = o.get("hit") or {}
        hit = h.get("type") == "block"
        rel, face = [0.0] * 3, [0.0] * 3
        if hit:
            b = h["block"]
            rx, rz = rotate(b[0] + 0.5 - x, b[2] + 0.5 - z, k)
            rel = [rx / 3, (b[1] + 0.5 - y) / 3, rz / 3]
            lx, lz = rotate((h["loc"][0] - b[0] - 0.5) * 2, (h["loc"][2] - b[2] - 0.5) * 2, k)
            face = [lx, (h["loc"][1] - b[1] - 0.5) * 2, lz]
        solid = (np.frombuffer(o["solid"].encode(), dtype=np.uint8) - 48)[GRID[k % 4]]
        head = [
            np.clip(gx / 20, -2, 2), np.clip(gz / 20, -2, 2), (Y + 1 + self.dy - y) / 5,
            min(math.hypot(gx, gz) / 20, 2.0), side, ahead,
            fx, fz, y - math.floor(y) - 0.5,
            cvx * 5, vy * 2, cvz * 5,
            residual, o["pitch"] / 90,
            float(o["on_ground"]), 0.0, float(hit),  # the 0.0 is the shift slot, filled in by _with_timer
            *rel, *face, min(o["held_count"] / 20, 2.0),  # same scale as the distance to the goal
            np.clip((self.par - self.used) / self.par, -2, 1),
            # The five below are replaced with current values in _with_timer: a player always knows them.
            0.0, 0.0, 0.0, 0.0, 0.0,
            self._edge_left(o), self._place_gain(o), min(self._walk(x, z) / 20, 2.0), *self._look_ahead(o),
        ]
        return np.concatenate([np.asarray(head, np.float32), solid.astype(np.float32)])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._use_task(self._pick_stage())
        spec = self.spec = self._spec()  # kept for whoever watches: the camera keeps clear of what it built
        self._prepare_arena()
        self.home = dict(pos=spec["pos"], yaw=spec["yaw"], pitch=spec["pitch"])
        self.obs = self.c.reset(**spec)
        self._after_reset()
        self.limit = self.time_limit()
        self._begin_episode()
        return self._seen(self._vec(self.obs)), {}

    def _after_reset(self):
        """Room for blocks a reset cannot build itself."""

    def _wait_dead(self):
        if self.realtime:
            next_tick = time.perf_counter()
            for _ in range(RESPAWN_TICKS):  # one tick at a time, so watchers see normal speed
                self.c.step()
                next_tick = pace(next_tick, 0.05)
        else:
            self.c.step(ticks=RESPAWN_TICKS)
        self.t += RESPAWN_TICKS

    def _respawn(self):
        """Dead for 4 seconds of game time, then back at the start with the bridge and blocks kept."""
        self._wait_dead()
        held = self.obs["held_count"]
        self.obs = self.c.reset(keep_arena=True, blocks=held, **self.home)
        self.delayed.clear()
        self.history.extend([np.zeros(8, np.float32)] * HISTORY)
        self._release_inputs()
        self.best = self._distance(self.obs)  # "stuck" now means no closer than since this respawn
        self.since_best = 0
        self.jumped = False
        self.delayed.append(self._vec(self.obs))

    def time_limit(self):
        """Ticks for this task: counted from the blocks it takes, so diagonals, height and detours are included."""
        return 60 + self.ticks_per_block * (max(self.par, self.walk_par) + 1 + max(0, -self.dy))

    def _release_inputs(self):
        self.keys = {k: False for k in KEYS}
        self.key_age = {k: MIN_KEY_TICKS for k in KEYS}
        self.late = set()
        self.late_click = False
        self.since_sneak = SNEAK_CLICK_TICKS + 1
        self.sneak_held = 0

    def _begin_episode(self):
        self.used = 0
        self.placed = {}  # block -> block it was placed against
        self.touched = set()
        self.t = 0
        self.best = self._distance(self.obs)
        self.ground_distance = self.best
        self.since_best = 0
        self.damage = 0.0
        self.delay = int(self.np_random.integers(self.delay_low, self.delay_high + 1))
        self.delayed = deque(maxlen=self.delay + 1)
        self.history = deque([np.zeros(8, np.float32)] * HISTORY, maxlen=HISTORY)
        self._release_inputs()
        self.sneak_ticks = 0
        self.jumped = False
        self.deaths = 0
        self.stopped_on = set()
        self.landed = 0
        self.parts = {}

    def _seen(self, vec):
        """What the agent gets: the game as it was a moment ago, plus what it has pressed since."""
        self.delayed.append(vec)
        return self._with_timer(np.concatenate([self.delayed[0], *self.history]))

    def _press(self, wanted):
        """Applies key limits: minimum press time, and a change can arrive one tick late."""
        if self.keys["sneak"]:
            self.sneak_held += 1
            if self.sneak_held >= self.max_sneak_ticks:
                wanted["sneak"] = False
                self.keys["sneak"] = False
                self.key_age["sneak"] = -1  # counts up to 0 below, so the release lasts the minimum press time
                self.late.discard("sneak")
                self.sneak_held = 0
        for key, want in wanted.items():
            self.key_age[key] += 1
            if want == self.keys[key]:
                self.late.discard(key)
                continue
            if self.key_age[key] < MIN_KEY_TICKS:
                continue
            if key not in self.late and self.np_random.random() < self.late_chance:
                self.late.add(key)
                continue
            self.late.discard(key)
            self.keys[key] = want
            self.key_age[key] = 0
            if key == "sneak":
                self.sneak_held = 0
        return dict(self.keys)

    def _click(self, wanted):
        """At most one click per tick, which is 20 per second. A click can also arrive one tick late."""
        if self.late_click:
            self.late_click = wanted
            return True
        if wanted and self.np_random.random() < self.late_chance:
            self.late_click = True
            return False
        return wanted

    def _aim(self, step):
        if step == 0.0:
            return 0.0
        return step + float(self.np_random.uniform(-self.aim_noise, self.aim_noise))

    def step(self, action):
        reward, term, trunc, info = 0.0, False, False, {}
        move, strafe, sneak, use, jump, sprint, yaw, pitch = (int(a) for a in action)
        self.history.append(np.asarray(
            [move / 2, strafe / 2, sneak, use / 2, jump, sprint, yaw / (self.turn_choices - 1), pitch / (self.turn_choices - 1)], np.float32))
        for tick in range(self.ticks_per_decision):
            r, term, trunc, info = self._tick(move, strafe, sneak, use > tick, jump, sprint, yaw, pitch)
            reward += r
            if info["respawned"]:
                break
            self.delayed.append(self._vec(self.obs))
            if term or trunc:
                break
        return self._with_timer(np.concatenate([self.delayed[0], *self.history])), reward, term, trunc, info

    def _points(self, rule, value):
        self.parts[rule] = self.parts.get(rule, 0.0) + value * self.reward_scale
        return value

    def _tick(self, move, strafe, sneak, use, jump, sprint, yaw, pitch):
        phase = STAGES[self.stage][8]
        helpers = phase == LEARN and self.extras == JUMP
        efficient = phase == EFFICIENT
        was_on_ground = self.obs["on_ground"]
        aim_before = self._way_in_view(self.obs)[0] if helpers else 0.0
        held = self.obs["held_count"]
        held_health = self.obs["health"]
        held_pitch = self.obs["pitch"]
        keys = self._press(dict(
            forward=move == 1, back=move == 2, left=strafe == 1, right=strafe == 2,
            sneak=bool(sneak), jump=bool(jump), sprint=bool(sprint),
        ))
        self.since_sneak = 0 if keys["sneak"] else self.since_sneak + 1
        self.sneak_ticks += keys["sneak"]
        clicked = self._click(use) and self.since_sneak <= SNEAK_CLICK_TICKS
        self.obs = o = self.c.step(use=clicked, yaw=self._aim(YAW_STEPS[yaw]), pitch=self._aim(PITCH_STEPS[pitch]), **keys)
        self.t += 1
        now = self._distance(o)
        reward = self._points("time", -TICK_COST)
        if use and held == 0:
            reward += self._points("click without blocks", -(EMPTY_CLICK_COST))
        # Distance only counts once it is standing on something, so a jump into the void earns nothing back.
        if o["on_ground"]:
            gained = self.ground_distance - now
            if self.deaths == 0 or gained > 0:  # walking back after a respawn pays nothing until it gets further
                reward += self._points("progress", gained)
                self.ground_distance = now
        if self.landed < len(self.landing_order) and o["on_ground"]:
            if self._is_on(o, self.landing_order[self.landed]):
                self.landed += 1
                reward += self._points("landed on a block", LANDING_BONUS)
        if self.extras in STOP_COURSES and o["on_ground"] and math.hypot(o["vel"][0], o["vel"][2]) < 0.03:
            here = self._standing_on(o)
            if here is not None and here not in self.stopped_on:
                self.stopped_on.add(here)
                reward += self._points("stopped on a block", STOP_BONUS)
        if self.extras in NO_BLOCKS and keys["sneak"]:
            reward += self._points("shift on jump stage", -(JUMP_SHIFT_COST))
        if helpers:
            reward += self._points("aim helper", AIM_REWARD * (self._way_in_view(o)[0] - aim_before))
            left_ground = keys["jump"] and was_on_ground and not o["on_ground"] and o["vel"][1] > 0.2
            if left_ground and not self.jumped and self._at_edge_toward_goal(o):
                self.jumped = True
                reward += self._points("jump helper", FIRST_JUMP_BONUS)
        if now < self.best - 0.01:
            self.best = now
            self.since_best = 0
        else:
            self.since_best += 1
        for _ in range(max(0, held - o["held_count"])):
            self.used += 1
            if efficient:
                reward += self._points("blocks", -(BLOCK_COST if self.used <= self.par else WASTE_COST))
        if held > o["held_count"] and o.get("clicks"):
            c = o["clicks"][-1]
            self.placed[tuple(c[:3])] = tuple(c[3:])
            for i in range(3):  # a block built outside the arena must also be cleared next time
                self.dirty[i] = min(self.dirty[i], c[i])
                self.dirty[i + 3] = max(self.dirty[i + 3], c[i])
        self._touched(o)
        if efficient:
            no_effect = (clicked and o["held_count"] >= held and held > 0) + (keys["sprint"] and not keys["forward"])
            no_effect += (pitch < 3 and held_pitch <= -90.0) or (pitch > 3 and held_pitch >= 90.0)
            reward += self._points("inputs with no effect", -(NO_EFFECT_COST * no_effect))
        success = self.reached(o)
        fell = self.fell(o) and not success
        if success:
            # Stand still briefly, so damage from this landing counts before the episode ends.
            for _ in range(DAMAGE_SETTLE_TICKS):
                o = self.c.step()
            self.obs = o
        stalled = not (success or fell) and self.since_best >= STALL_TICKS
        lost = max(0.0, held_health - o["health"])
        reward += self._points("fall damage", -(0.5 * lost))
        self.damage += lost
        if success:
            reward += self._points("goal", GOAL_REWARD)
            if efficient:
                reward += self._points("speed", SPEED_BONUS * (1.0 - self.t / self.limit))
                reward += self._points("block efficiency", EFFICIENCY_BONUS * self.par / max(self.used, self.par))
                reward += self._points("unused blocks", -(UNUSED_BLOCK_COST * self.unused_blocks()))
        if fell:
            reward += self._points("death", -(DEATH_COST[phase]))
        if stalled:
            reward += self._points("stuck", -(STALL_COST))
        respawned = False
        out_of_lives = False
        if fell:
            self.deaths += 1
            out_of_lives = self.deaths >= MAX_DEATHS
            if not out_of_lives:
                self._respawn()
                respawned = True
                o = self.obs
        # The time limit is part of the task and the model can see it, so it is a real end of the round.
        timed_out = self.t >= self.limit and not (success or stalled or out_of_lives)
        term = success or stalled or out_of_lives or timed_out
        if (stalled or timed_out) and not fell and self._on_start_platform(o):
            reward += self._points("stayed on the start", -(STAYED_HOME_COST))
        info = {"success": success, "fell": self.deaths > 0, "deaths": self.deaths, "respawned": respawned, "stalled": stalled,
                "damage": self.damage, "blocks_used": self.used, "blocks_needed": self.par, "unused": self.unused_blocks(),
                "shift_share": self.sneak_ticks / max(self.t, 1), "current": self.played_stage == self.stage, "parts": dict(self.parts)}
        if success:
            info["blocks_per_s"] = math.hypot(self.goal_x, self.goal_z) / (self.t / 20)
        return float(reward) * self.reward_scale, term, False, info
