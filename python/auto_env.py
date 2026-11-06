"""Every course from one generator, seen from the player's own view, with one set of point rules.

Uses the input limits and course parts of SkillEnv. What differs: curriculum.py picks the round and nothing is lost
while dead in training.

What the model sees has to work in normal Minecraft too, so nothing here tells it the way or the rules of a course.
Two frames are used. Blocks, position in the block and aim are in the block grid, turned in steps of 90 degrees so that
"ahead" is the main direction of the goal. That frame stays still during a round: turning the head changes nothing in the
block view, and block edges stay exact. Movement and the goal are relative to where the player looks, because that is how
the keys work.
"""
import math
import time

import gymnasium as gym
import numpy as np

import bedwars_map
import curriculum
from jump_physics import hits_pole, jump_possible
from skill_env import (BLOCK_COST, FACES, DAMAGE_SETTLE_TICKS, EMPTY_CLICK_COST, GOAL_REWARD, HISTORY, HOPS, LANDING_BONUS, MAX_DEATHS,
                       POLE_HEIGHT, RESPAWN_TICKS, RUN, RUN_HOLES, SNEAK_CLICK_TICKS, STALL_TICKS, STAYED_HOME_COST, TICK_COST, TREENAN,
                       WASTE_COST, Y, SkillEnv, pace, rotate)

RADIUS = 5  # blocks the model sees in every direction: ahead, behind, to the sides, up and down
_size = 2 * RADIUS + 1
VIEW_SIZE = _size ** 3
_view = np.arange(-RADIUS, RADIUS + 1)
_x, _up, _z = (a.ravel() for a in np.meshgrid(_view, _view, _view, indexing="ij"))  # in the frame of the view values
HEAD = 4 + 3 + 3 + 3 + 6 + 7 + 6 + 1 + 1  # goal, place in block, speed, look, body, aim, side aimed at, blocks, can place
KNOWN = 9  # values a player always knows: time left, the keys it is holding and the wait before it can jump again
MAP_RADIUS = 16  # columns in every direction on the map around the player
MAP_SIDE = 2 * MAP_RADIUS + 1
MAP_CHANNELS = 3  # height of the top block, whether a block is there at all, whether it has stood there this round
MAP_ABOVE, MAP_BELOW = 8, 16  # how far a column is searched in the game
_map = np.arange(-MAP_RADIUS, MAP_RADIUS + 1)
_map_x, _map_z = (a.ravel() for a in np.meshgrid(_map, _map, indexing="ij"))
TURN_STEPS = (-60.0, -15.0, -5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0, 15.0, 60.0)  # degrees per tick: fine near zero, like a hand
NO_TURN = TURN_STEPS.index(0.0)
JUMP_WAIT = 10  # ticks the game makes it wait between jumps while the jump key stays down
DEATH_COST = 4.0
STUCK_COST = DEATH_COST  # giving up must not be cheaper than trying and falling
GOAL_AFTER_DEATH = 0.5  # share of the goal points left when it took a second try
SPEED_BONUS = 3.0  # at the goal, by the share of the time limit left
PLACE_BONUS = 1.0  # per block of reach closer to the goal than ever, by a block placed or a block stood on: never paid twice
LOOK_UP_COST = 0.05  # per tick when looking up at nothing, at the sky, in proportion to how far above level it looks
RECORD_GRACE = 20  # ticks of spare time when chasing speed, enough to turn around at the start
FACES_3D = {"north": (0, 0, -1), "south": (0, 0, 1), "west": (-1, 0, 0), "east": (1, 0, 0), "up": (0, 1, 0), "down": (0, -1, 0)}
FOUR = ((1, 0), (0, 1), (-1, 0), (0, -1))
MAX_HEIGHT_CHANGE = 10  # above or below the start, on long courses
FAR_JUMP = (4, 3)  # just inside what the jump math allows from a block
SLAB = "minecraft:stone_slab[type=bottom,waterlogged=false]"
MOVED_ON = 1.5  # blocks from where it last was to count as going somewhere
DETOUR_TICKS = 200  # longest it may go around something without getting closer: 10 s
CLUTCH_BONUS = 2.0  # back on a block after a push, without falling
SAVED_BLOCK_BONUS = 0.3  # at the goal, per block under the estimated need: as much as a block over it costs (WASTE_COST)
PUSH_FALL_MARGIN = 4.0  # with pushes a clutch lands below the bridge, so a fall counts a little further down
PUSH_WAIT = 60  # ticks a planned push waits for a moment it fits, then it is dropped
PUSH_SPARE_BLOCKS = 4  # handed out per planned push, beyond the need: a save and the way back up
PUSH_SPARE_TICKS = 60  # and time for it


CUBE_START = HEAD + KNOWN
CUBE_SIZE = VIEW_SIZE
MAP_START = CUBE_START + CUBE_SIZE
HISTORY_START = MAP_START + MAP_CHANNELS * MAP_SIDE ** 2


class AutoEnv(SkillEnv):
    turn_choices = len(TURN_STEPS)
    ticks_per_decision = 1  # exact jumps and fast bridges need every tick
    max_sneak_ticks = float("inf")  # shift stays down for as long as it is held
    N_OBS = HISTORY_START + 8 * HISTORY
    visited = frozenset()
    delay_low, delay_high, late_chance, aim_noise = 0, 0, 0.0, 0.0  # it sees the game as it is now, and keys and turns land exactly
    family = None
    task = None
    tested = None
    mastered = ()  # course kinds that need less practice
    jump_wait, was_on_ground = 0, True
    exam_family = None  # plays only this kind: with every dial at its limit for a test, or as in training for watching
    exam_all = True
    can_push = True
    pushes = ()

    def __init__(self, port, levels=None, render=False):
        self.levels = levels or curriculum.Curriculum().levels
        super().__init__(port, stage=0, render=render)
        # A click is always one click, never a held button: click or not, once per decision.
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2, 2, 2, 2, len(TURN_STEPS), len(TURN_STEPS)])

    def set_stage(self, stage):
        pass

    practice = None  # how much each course kind is played, from curriculum.Curriculum.practice

    def set_levels(self, levels, mastered=(), practice=None):
        self.levels, self.mastered, self.practice = levels, mastered, practice

    def set_exam(self, family, all_dials=True):
        self.exam_family, self.exam_all = family, all_dials

    def reset(self, *, seed=None, options=None):
        """A very long course can reach past what the game has loaded yet. Then another round is drawn instead."""
        for _ in range(4):
            try:
                return super().reset(seed=seed, options=options)
            except RuntimeError as e:
                if not any(k in str(e) for k in ("did not settle", "no bedwars map", "no reachable")):
                    raise
                print(f"round skipped: {e}", flush=True)
                seed = None
        return super().reset(options=options)

    # ---- Choosing the round ----

    def _pick_stage(self):
        family = self.exam_family or curriculum.pick_family(self.levels, self.np_random, self.mastered, self.practice)
        settings, tested = curriculum.draw(self.levels, family, self.np_random, "all" if self.exam_family and self.exam_all else None)
        return family, settings, tested

    def _use_task(self, picked):
        self.family, self.task, self.tested = picked
        t = self.task
        self.shape_cells = {}  # blocks of other shapes than a full block, put in after the reset built the course
        self.map_kind = "treenan"
        self.goal_size = t.get("goal", "platform")
        self.kind, self.blocked, self.detour, self.field = None, set(), 0, None
        self.goal_half, self.field_margin, self.fall_margin, self.walk_par = 0, 6, 1.5, 0
        self.ticks_per_block, self.budget = t.get("faster") or t["ticks"], t.get("budget", (0.0, 0))
        self.landing_order, self.course, self.poles = [], [], []
        self.landed = 0
        # Pushes: when in the round (a share of the time limit), which way and how hard. Placed in time once the limit is known.
        level = curriculum.PUSHES.index(t.get("push"))
        r = self.np_random
        self.push_draws = [(float(r.uniform(0.1, 0.6)), float(r.random()), float(r.uniform(0, 22.5))) for _ in range(int(r.integers(1, t.get("hits", 2) + 1)))] if level else []
        self.push_level = level
        self.push_spare = PUSH_SPARE_BLOCKS * len(self.push_draws)
        self.gap, self.heights, self.directions, self.yaw_range, self.start = 8, (0,), "axis", t.get("yaw", 180), "center"
        if self.family == "hop":
            self.extras = HOPS
        elif self.family == "path":
            self.extras = RUN_HOLES if t["holes"] else RUN
        elif self.family == "bridge":
            self.extras = t["obstacle"]
            self.gap = int((max(t["gap"], 8) if t["obstacle"] else t["gap"]) * t.get("longer", 1.0))
            self.heights, self.directions, self.start = t["heights"], t["directions"], t["start"]
        elif self.family == "stairs":
            # The goal as many blocks up as the gap is long: 45 degrees, one step up per block built ahead.
            self.extras = None
            down = t.get("way") == "up or down" and self.np_random.random() < 0.5
            self.gap = max(2, round(t["rise"] / t.get("steep", 1.0)))
            self.heights, self.directions, self.start = (-t["rise"] if down else t["rise"],), t["directions"], t["start"]
        elif self.family == "bedwars":
            # Played like the Treenan map: blocks pasted in before the round, a way around what stands in it.
            self.extras = TREENAN
            self.map_kind = "bedwars"
            self.goal_half, self.field_margin, self.fall_margin = 0, 24, 6.0
        else:
            self.extras = TREENAN
            self.goal_half, self.field_margin, self.fall_margin = 0, 18, 6.0
            self.map_gaps = (10, t["gap"])
            self.map_rise = t.get("height", 0)
            self.map_start = t.get("start", "bed")
            self.map_spots = t.get("spots", "grass")

    def _spec(self):
        spec = self._hop_spec() if self.family == "hop" else self._path_spec() if self.family == "path" else super()._spec()
        if self.push_spare and spec.get("blocks", 0) > 0:
            spec["blocks"] = self.blocks_given = min(64, spec["blocks"] + self.push_spare)
        spec["block_radius"] = RADIUS
        spec["map_radius"] = MAP_RADIUS
        return spec

    def _hop_spec(self):
        """Single blocks in a row. Each is possible by the jump math from a block with just a short run-up."""
        t, r = self.task, self.np_random
        main = FOUR[int(r.integers(4))]
        on_platform = t["takeoff"] == "platform"
        x, z, height = (main[0], main[1], Y) if on_platform else (0, 0, Y)
        offsets = [t["offset"]]
        ladder = curriculum.FAMILIES["hop"]["offset"]
        if t.get("mix"):
            offsets = list(ladder[: ladder.index(t["offset"]) + 1])
        if t.get("far"):
            offsets.append(FAR_JUMP)
        sign = 1 if r.random() < 0.5 else -1
        blocks, poles = [], []
        for _ in range(t["count"] * t.get("longer", 1)):
            ahead, sideways = offsets[int(r.integers(len(offsets)))]
            if t.get("bend") and r.random() < 0.3:
                main = (main[1], main[0]) if r.random() < 0.5 else (-main[1], -main[0])
            # The first way that keeps clear of the course so far. With nowhere left, the course ends here.
            ways = [(h, g) for h in (main, (main[1], main[0]), (-main[1], -main[0])) for g in (sign, -sign)]
            way = next((w for w in ways if self._hop_is_free(x, z, w[0], ahead, w[1] * sideways, blocks, 2 if on_platform else 0)), None)
            if way is None:
                break
            heading, sign = way
            main = heading
            step = int(r.choice(t["height"]))
            if not jump_possible(ahead, sideways, step, run_up_ticks=2) or abs(height + step - Y) > MAX_HEIGHT_CHANGE:
                step = 0
            side = (heading[1], heading[0])
            if abs(sideways) >= 2:
                poles.append(((x + heading[0] * ahead, z + heading[1] * ahead), height))  # in the row it leaves
            x += heading[0] * ahead + side[0] * sign * sideways
            z += heading[1] * ahead + side[1] * sign * sideways
            height += step
            blocks.append((x, height, z))
            sign = -sign
        self.course = blocks
        slabs = self._hop_slabs(blocks) if t.get("shapes") else set()
        # On a slab the feet are half a block lower, inside the slab's own block.
        self.landing_order = [(bx, by if (bx, by, bz) in slabs else by + 1, bz) for bx, by, bz in blocks]
        self.goal_x, self.goal_z, self.dy = x, z, height - Y
        self.par, self.blocks_given = len(blocks), 0
        self.poles = self._safe_poles(poles)
        boxes = [[bx, by, bz, bx, by, bz] for bx, by, bz in blocks if (bx, by, bz) not in slabs]
        boxes += [[px, py - 1, pz, px, py + POLE_HEIGHT, pz] for px, py, pz in self.poles]
        return self._course_spec(boxes, on_platform)

    def _hop_slabs(self, blocks):
        """Some blocks become bottom slabs: landing on one is half a block lower, and jumping from it half a block
        harder. Only where the jump off it still passes the jump math."""
        slabs = set()
        for i in range(1, len(blocks)):
            b = blocks[i]
            if self.np_random.random() >= 0.35:
                continue
            if i + 1 < len(blocks):
                n = blocks[i + 1]
                dx, dz = abs(n[0] - b[0]), abs(n[2] - b[2])
                if not jump_possible(max(dx, dz), min(dx, dz), n[1] - b[1] + 0.5, run_up_ticks=2):
                    continue
            slabs.add(b)
            self.shape_cells[b] = SLAB
        return slabs

    @staticmethod
    def _hop_is_free(x, z, heading, ahead, sideways, blocks, start_size):
        side = (heading[1], heading[0])
        nx, nz = x + heading[0] * ahead + side[0] * sideways, z + heading[1] * ahead + side[1] * sideways
        if max(abs(nx), abs(nz)) <= start_size:
            return False
        if not all(max(abs(nx - b[0]), abs(nz - b[2])) > 1 for b in blocks):
            return False
        # The jump itself: a block near its line, at any height, could catch the head or the feet on the way over.
        others = [b for b in blocks if (b[0], b[2]) != (x, z)]
        for i in range(1, 10):
            px, pz = x + (nx - x) * i / 10, z + (nz - z) * i / 10
            if any(abs(px - b[0]) < 1.1 and abs(pz - b[2]) < 1.1 for b in others):
                return False
        # Nor does the new block stand in the way of a jump made before it.
        way = [(0, 0)] + [(b[0], b[2]) for b in blocks]
        for (ax, az), (bx, bz) in zip(way[:-1], way[1:]):
            for i in range(1, 10):
                px, pz = ax + (bx - ax) * i / 10, az + (bz - az) * i / 10
                if abs(px - nx) < 1.1 and abs(pz - nz) < 1.1:
                    return False
        return True

    def _course_spec(self, boxes, on_platform):
        """The round's start, arena box and camera, for courses built from single blocks."""
        r = self.np_random
        blocks = self.course
        lowest = min(Y + 1, min(b[1] for b in blocks) + 1)
        self.fall_margin = 1.5 + min(Y + 1, Y + 1 + self.dy) - lowest
        boxes = [[-1, Y, -1, 1, Y, 1] if on_platform else [0, Y, 0, 0, Y, 0], *boxes]
        extent = boxes + [[x, y, z, x, y, z] for x, y, z in self.shape_cells]  # slabs are part of the arena too
        box = [min(b[0] for b in extent) - 4, min(b[1] for b in extent) - 3, min(b[2] for b in extent) - 4,
               max(b[3] for b in extent) + 4, max(b[4] for b in extent) + POLE_HEIGHT + 3, max(b[5] for b in extent) + 4]
        clear = [min(a, b) for a, b in zip(self.dirty[:3], box[:3])] + [max(a, b) for a, b in zip(self.dirty[3:], box[3:])]
        self.dirty = box
        spread = 0.3 if on_platform else 0.1
        px, pz = 0.5 + float(r.uniform(-spread, spread)), 0.5 + float(r.uniform(-spread, spread))
        first = blocks[0]
        yaw = math.degrees(math.atan2(-(first[0] + 0.5 - px), first[2] + 0.5 - pz)) + float(r.uniform(-self.yaw_range, self.yaw_range))
        pitch = float(r.uniform(-5, 15)) if self.yaw_range <= 15 else float(r.uniform(-30, 60))
        gy = Y + self.dy
        return dict(clear=clear, camera=self.camera_for(self.goal_x, self.goal_z, gy), camera_every=self.camera_every,
                    beam=[self.goal_x + 0.5, gy + 1.2, self.goal_z + 0.5], platforms=boxes, pos=[px, Y + 1, pz],
                    yaw=float(yaw % 360), pitch=pitch, blocks=0, block_radius=RADIUS)

    def _path_cells(self):
        """A one block wide way, in (ahead, sideways, height). Never touches itself, so it cannot be cut short."""
        t, r = self.task, self.np_random
        length = t["length"] * t.get("longer", 1)
        wind, slope = t.get("wind", "gentle"), t.get("slope", "flat")
        longest_hole = 3 if t.get("wide_holes") else 2
        step_chance, step_gap = {"flat": (0.0, 0), "some steps": (0.25, 3), "stairs": (0.5, 1), "stairs and drops": (0.5, 1)}[slope]
        for _ in range(40):
            cells, holes = [(2, 0, Y)], []
            heading = (1, 0)
            straight, since_step, next_turn = 1, 0, 2
            ok = True

            def free(a, s):
                if a < 2 or abs(s) > 5:
                    return False
                if a == 2 and abs(s) <= 2:
                    return False
                return all(max(abs(a - c[0]), abs(s - c[1])) > 1 for c in cells[:-2]) and (a, s) not in {c[:2] for c in cells[-2:]} and (a, s) not in holes

            while len(cells) < length and ok:
                a, s, h = cells[-1]
                left = length - len(cells)
                ahead_free = all(free(a + heading[0] * i, s + heading[1] * i) for i in range(1, longest_hole + 3))
                if t.get("holes") and straight >= 3 and left >= 3 and ahead_free and r.random() < 0.35:
                    size = int(r.integers(1, longest_hole + 1))
                    holes += [(a + heading[0] * i, s + heading[1] * i) for i in range(1, size + 1)]
                    for i in (size + 1, size + 2):
                        cells.append((a + heading[0] * i, s + heading[1] * i, h))
                    straight, since_step = 2, since_step + 2
                    continue
                if straight >= next_turn:
                    if heading == (1, 0):
                        turns = [(0, 1), (0, -1)]
                    else:
                        turns = [(1, 0)] + ([(-1, 0)] if wind == "around" else [])
                    turns = [d for d in turns if free(a + d[0], s + d[1])]
                    if turns:
                        heading = turns[int(r.integers(len(turns)))]
                        straight = 0
                        next_turn = int(r.integers(2, 5)) if wind == "gentle" else int(r.integers(1, 3))
                na, ns = a + heading[0], s + heading[1]
                if not free(na, ns):
                    options = [d for d in ((1, 0), (0, 1), (0, -1), (-1, 0)) if free(a + d[0], s + d[1]) and (wind == "around" or d != (-1, 0))]
                    if not options:
                        ok = False
                        break
                    heading = options[int(r.integers(len(options)))]
                    straight = 0
                    na, ns = a + heading[0], s + heading[1]
                nh = h
                since_step += 1
                if since_step >= step_gap and step_chance and r.random() < step_chance:
                    change = int(r.choice((-2, -1, 1))) if slope == "stairs and drops" else int(r.choice((-1, 1)))
                    if abs(h + change - Y) <= 6:
                        nh = h + change
                        since_step = 0
                cells.append((na, ns, nh))
                straight += 1
            if not ok:
                continue
            # The goal platform takes up 5 by 5 ahead of the last block, and only that block may touch it.
            a, s, _ = cells[-1]
            clear_of_goal = all(c[0] < a or abs(c[1] - s) > 3 or c == cells[-1] for c in cells) and all(h_[0] < a + 1 or abs(h_[1] - s) > 3 for h_ in holes)
            if clear_of_goal:
                return cells, holes
        return [(2 + i, 0, Y) for i in range(length)], []

    def _path_spec(self):
        r = self.np_random
        main = FOUR[int(r.integers(4))]
        side = (main[1], main[0])
        world = lambda a, s: (main[0] * a + side[0] * s, main[1] * a + side[1] * s)
        cells, holes = self._path_cells()
        self.course = [(world(a, s)[0], h, world(a, s)[1]) for a, s, h in cells]
        last_a, last_s, last_h = cells[-1]
        self.goal_x, self.goal_z = world(last_a + 3, last_s)
        self.dy = last_h - Y
        edge_x, edge_z = world(last_a + 1, last_s)
        sunk = self._path_shapes() if self.task.get("shapes") else set()
        self.landing_order = [(x, y if (x, y, z) in sunk else y + 1, z) for x, y, z in self.course] + [(edge_x, last_h + 1, edge_z), (self.goal_x, last_h + 1, self.goal_z)]
        self.par, self.blocks_given = len(cells), 0
        boxes = [[x, y, z, x, y, z] for x, y, z in self.course if (x, y, z) not in sunk]
        boxes.append([self.goal_x - 2, last_h, self.goal_z - 2, self.goal_x + 2, last_h, self.goal_z + 2])
        return self._course_spec(boxes, True)

    def _path_shapes(self):
        """Slabs as on real maps: a half step on the block before a step up, so it can be walked without a jump, and
        some blocks sunk to a slab, half a block lower. Never next to a hole, where the jump has to stay as it is.
        Returns the sunk blocks."""
        r, course, sunk = self.np_random, self.course, set()
        near = lambda a, b: max(abs(a[0] - b[0]), abs(a[2] - b[2])) == 1
        for i in range(1, len(course) - 1):
            p, c, n = course[i - 1], course[i], course[i + 1]
            if not (near(p, c) and near(c, n)):
                continue
            # A half step only where it comes from the same height or from above: from below it would be 1.5 up.
            if n[1] == c[1] + 1 and p[1] >= c[1] and r.random() < 0.6:
                self.shape_cells[(c[0], c[1] + 1, c[2])] = SLAB
            elif p[1] == c[1] == n[1] and (p not in sunk) and r.random() < 0.25:
                sunk.add(c)
                self.shape_cells[c] = SLAB
        if not self._path_possible(sunk):  # never a course that cannot be done: then no slabs at all
            self.shape_cells.clear()
            sunk = set()
        return sunk

    def _path_possible(self, sunk):
        """Every step of the path, with its slabs, can be walked or jumped: up at most one block between blocks next
        to each other, and every jump over a hole within the jump math."""
        def top(cell):
            x, h, z = cell
            if cell in sunk:
                return h + 0.5
            return h + 1.5 if (x, h + 1, z) in self.shape_cells else h + 1.0

        way = [(0, Y, 0)] + list(self.course)  # from the start platform's middle onward
        for a, b in zip(way, way[1:]):
            dx, dz = abs(a[0] - b[0]), abs(a[2] - b[2])
            rise = top(b) - (Y + 1.0 if a == (0, Y, 0) else top(a))
            if max(dx, dz) <= 1 or a == (0, Y, 0):
                if rise > 1.0:
                    return False
            elif not jump_possible(max(dx, dz), min(dx, dz), rise, run_up_ticks=2):
                return False
        return True

    def _bedwars_spec(self):
        """A fresh Bedwars map (bedwars_map.py), played like the Treenan map: pasted in, then from the start to the goal."""
        r = self.np_random
        for _ in range(20):
            m = bedwars_map.layout(self.task, r, Y, most_blocks=64 - self.push_spare)
            gx, gy, gz = m.goal
            self.goal_x, self.goal_z, self.dy = gx, gz, gy - Y
            self.blocked = set(m.blocked) - {(x, z) for x in range(-1, 2) for z in range(-1, 2)}
            self._build_field()
            if self.field is None or self._next_cell(0.5, 0.5)[1] is not None and self._walk(0.5, 0.5) < self.field.max() - 1.0:
                break  # the goal can be reached on foot and by building, around everything in the way
        else:
            raise RuntimeError("no reachable bedwars map found")
        self.par = m.par
        self.walk_par = int(self._walk(0.5, 0.5))
        self.blocks_given = min(64, math.ceil(self.par * self.budget[0]) + self.budget[1])
        self.next_cells = m.cells
        return dict(clear=[0, 0, 0, -1, -1, -1], platforms=[], skip_block_check=True, camera=self.camera_for(gx, gz, gy), camera_every=self.camera_every,
                    beam=[gx + 0.5, gy + 1.2, gz + 0.5], pos=[0.5 + float(r.uniform(-0.3, 0.3)), Y + 1, 0.5 + float(r.uniform(-0.3, 0.3))],
                    yaw=float(r.uniform(0, 360)), pitch=float(r.uniform(-30, 89)), blocks=self.blocks_given, block_radius=RADIUS)

    def _plan_pushes(self):
        """The round's pushes as (from tick, until tick, strongest level, which way, compass offset). Needs the time limit."""
        self.pushes = []
        if not self.push_draws:
            return
        self.fall_margin = max(self.fall_margin, PUSH_FALL_MARGIN)
        for share, turn, jitter in sorted(self.push_draws):
            start = 20 + int(share * self.limit)
            if self.pushes:
                start = max(start, self.pushes[-1][1] + 40)  # one at a time
            self.pushes.append((start, start + PUSH_WAIT, self.push_level - 1, turn, jitter))

    def _safe_poles(self, poles):
        """Poles never stand in the way of a real jump, or on a block of the course."""
        solid = {(b[0], b[2]) for b in self.course} | {(x, z) for x in range(-1, 2) for z in range(-1, 2)}
        route = [(0.5, 0.5)] + [(b[0] + 0.5, b[2] + 0.5) for b in self.course]
        kept = []
        for (x, z), height in poles:
            if (x, z) in solid:
                continue
            if any(hits_pole(p, q, [(x, z)]) for p, q in zip(route, route[1:])):
                continue
            kept.append((x, height, z))
        return kept

    def time_limit(self):
        if self.task.get("faster"):
            return RECORD_GRACE + self.ticks_per_block * (self.par + 1 + max(0, -self.dy)) + PUSH_SPARE_TICKS * len(self.push_draws)
        return super().time_limit() + PUSH_SPARE_TICKS * len(self.push_draws)

    # ---- Where to go ----

    def _next_target(self, offset=0):
        i = min(self.landed + offset, len(self.landing_order) - 1)
        x, y, z = self.landing_order[i]
        return x + 0.5, y, z + 0.5

    def _route(self, x, z):
        if not self.landing_order:
            return super()._route(x, z)
        tx, _, tz = self._next_target()
        norm = math.hypot(tx - x, tz - z) or 1.0
        return (tx - x) / norm, (tz - z) / norm

    @staticmethod
    def _climb(ahead, up):
        """Distance left with `ahead` to go on the level and `up` to climb. With the goal higher up, getting closer
        on the level than the climb left gains nothing: the way up has to start before it, not under the goal."""
        return max(ahead, up) + up if up > 0 else ahead + abs(up)

    def _distance(self, o):
        if not self.landing_order:
            x, y, z = o["pos"]
            return self._climb(self._walk(x, z), Y + 1 + self.dy - y)
        x, y, z = o["pos"]
        tx, ty, tz = self._next_target()
        total = math.hypot(tx - x, tz - z) + abs(ty - y)
        for a, b in zip(self.landing_order[self.landed :], self.landing_order[self.landed + 1 :]):
            total += math.hypot(a[0] - b[0], a[2] - b[2]) + abs(a[1] - b[1])
        return total

    def reached(self, o):
        if self.family == "hop":
            return self.landed >= len(self.landing_order)
        return super().reached(o)

    # ---- What it sees ----

    def _edge_left(self, o):
        """How far along the way to go the ground under the feet continues, up to 2 blocks."""
        solid, feet = o.get("solid"), o.get("feet")
        if not solid or not feet:
            return 1.0
        x, _, z = o["pos"]
        rx, rz = self._route(x, z)
        size = round(len(solid) ** (1 / 3))  # older recordings hold a smaller cube
        radius = size // 2
        for i in range(21):
            d = i * 0.1
            cx, cz = int(math.floor(x + rx * d) - feet[0]), int(math.floor(z + rz * d) - feet[2])
            if solid[(cx + radius) * size * size + (radius - 1) * size + (cz + radius)] != "1":
                return d / 2
        return 1.0

    def _view(self, o, k):
        """Solid or not all around the player, exactly on the block grid, in the frame of this round."""
        solid = np.frombuffer(o["solid"].encode(), dtype=np.uint8) - 48
        size = round(solid.size ** (1 / 3))  # recordings hold a bigger cube than the model sees
        radius = size // 2
        x, z = _x, _z
        for _ in range(-k % 4):  # from the frame back to the world
            x, z = z, -x
        return solid[(x + radius) * size * size + (_up + radius) * size + (z + radius)].astype(np.float32)

    def _surface(self, o):
        """Top block per column, from the game, or worked out from the blocks in older recordings. None is nothing known."""
        text = o.get("surface")
        if text:
            codes = np.frombuffer(text.encode(), dtype=np.uint8)
            present = codes != 46
            side = round(codes.size ** 0.5)
            return (codes.astype(np.float32) - 65 - MAP_BELOW).reshape(side, side), present.reshape(side, side)
        solid = np.frombuffer(o["solid"].encode(), dtype=np.uint8) - 48
        size = round(solid.size ** (1 / 3))
        cube = solid.reshape(size, size, size).astype(bool)
        heights = np.arange(size, dtype=np.float32) - size // 2
        present = cube.any(axis=1)
        top = size - 1 - np.argmax(cube[:, ::-1, :], axis=1)
        return heights[top], present

    def _map_layers(self, o, k):
        """Height, block there, and visited, around the player in the frame of this round."""
        heights, present = self._surface(o)
        radius = heights.shape[0] // 2
        x, z = _map_x, _map_z
        for _ in range(-k % 4):  # from the frame back to the world
            x, z = z, -x
        inside = (np.abs(x) <= radius) & (np.abs(z) <= radius)
        height, there, been = np.zeros(_map_x.size, np.float32), np.zeros(_map_x.size, np.float32), np.zeros((MAP_SIDE, MAP_SIDE), np.float32)
        height[inside] = heights[x[inside] + radius, z[inside] + radius] / 8
        there[inside] = present[x[inside] + radius, z[inside] + radius]
        height *= there
        fx, fz = o["feet"][0], o["feet"][2]
        for vx, vz in self.visited:
            rx, rz = rotate(vx - fx, vz - fz, k)
            if abs(rx) <= MAP_RADIUS and abs(rz) <= MAP_RADIUS:
                been[int(rx) + MAP_RADIUS, int(rz) + MAP_RADIUS] = 1.0
        return np.concatenate([height, there, been.ravel()])

    def _visit(self, o):
        if o["on_ground"]:
            x, _, z = o["pos"]
            self.visited.add((int(math.floor(x)), int(math.floor(z))))

    def _vec(self, o):
        x, y, z = o["pos"]
        vx, vy, vz = o["vel"]
        turn = math.radians(o["yaw"])
        forward, right = (-math.sin(turn), math.cos(turn)), (-math.cos(turn), -math.sin(turn))
        ego = lambda dx, dz: (dx * forward[0] + dz * forward[1], dx * right[0] + dz * right[1])
        # The frame of this round: the goal does not move, so neither does this.
        k = int(round(math.degrees(math.atan2(-self.goal_x, self.goal_z)) / 90.0))
        goal_x, goal_z = self.goal_x + 0.5 - x, self.goal_z + 0.5 - z
        distance = math.hypot(goal_x, goal_z)
        goal_f, goal_r = ego(goal_x, goal_z)
        goal_up = Y + 1 + self.dy - y
        centre_x, centre_z = rotate(x - math.floor(x) - 0.5, z - math.floor(z) - 0.5, k)
        speed_f, speed_r = ego(vx, vz)
        look = turn - k * math.pi / 2
        h = o.get("hit") or {}
        hit = h.get("type") == "block"
        rel, face = [0.0] * 3, [0.0] * 3
        if hit:
            b = h["block"]
            hx, hz = rotate(b[0] + 0.5 - x, b[2] + 0.5 - z, k)
            rel = [hx / 3, (b[1] + 0.5 - y) / 3, hz / 3]
            lx, lz = rotate((h["loc"][0] - b[0] - 0.5) * 2, (h["loc"][2] - b[2] - 0.5) * 2, k)
            face = [lx, (h["loc"][1] - b[1] - 0.5) * 2, lz]
        head = [
            goal_f / max(distance, 0.01), goal_r / max(distance, 0.01), math.log1p(distance) / 4, math.copysign(math.log1p(abs(goal_up)), goal_up) / 3,
            centre_x, centre_z, y - math.floor(y) - 0.5,
            speed_f * 5, speed_r * 5, vy * 2,
            o["pitch"] / 90, math.sin(look), math.cos(look),
            float(o["on_ground"]), float(o.get("sprinting", False)), o.get("health", 20.0) / 20,
            float(o.get("collision", False)), math.log1p(o.get("fall_distance", 0.0)) / 3, o.get("hurt_time", 0) / 10,
            float(hit), *rel, *face, *self._side(h, k),
            min(o["held_count"] / 20, 2.0), float(self._can_place(o)),
        ]
        known = [0.0] * KNOWN  # filled in by _with_timer
        return np.clip(np.concatenate([np.asarray(head, np.float32), np.asarray(known, np.float32), self._view(o, k), self._map_layers(o, k)]), -5.0, 5.0)

    @staticmethod
    def _can_place(o):
        """Would a click put a block down right now? It has a block, aims at a block in reach, the spot next to that side
        is empty, and the player is not standing in it. Our own shift rule is not part of this: the model knows its keys."""
        h = o.get("hit") or {}
        if h.get("type") != "block" or o.get("held_count", 0) <= 0 or h.get("face") not in FACES_3D:
            return False
        step = FACES_3D[h["face"]]
        target = [int(h["block"][i]) + step[i] for i in range(3)]
        solid = o["solid"]
        size = round(len(solid) ** (1 / 3))
        radius = size // 2
        cell = [target[i] - int(o["feet"][i]) for i in range(3)]
        if max(abs(c) for c in cell) <= radius and solid[(cell[0] + radius) * size * size + (cell[1] + radius) * size + (cell[2] + radius)] == "1":
            return False
        x, y, z = o["pos"]
        height = 1.5 if o.get("sneaking", o.get("sneak")) else 1.8
        overlaps = (target[0] < x + 0.3 and x - 0.3 < target[0] + 1 and target[2] < z + 0.3 and z - 0.3 < target[2] + 1 and target[1] < y + height and y < target[1] + 1)
        return not overlaps

    @staticmethod
    def _side(h, k):
        """The side of the block it aims at, in the frame of this round: up, down, ahead, behind, one side, the other."""
        sides = [0.0] * 6
        name = h.get("face")
        if h.get("type") != "block" or name is None:
            return sides
        if name in ("up", "down"):
            sides[0 if name == "up" else 1] = 1.0
        else:
            x, z = rotate(*FACES[name], k)
            sides[2 if x > 0 else 3 if x < 0 else 4 if z > 0 else 5] = 1.0
        return sides

    def _track_jump(self, o, jump_key):
        """The game makes it wait between jumps only while the key stays down."""
        if not jump_key:
            self.jump_wait = 0
        elif self.jump_wait > 0:
            self.jump_wait -= 1
        if jump_key and self.was_on_ground and not o["on_ground"] and o["vel"][1] > 0.2:
            self.jump_wait = JUMP_WAIT
        self.was_on_ground = o["on_ground"]

    def _release_inputs(self):
        super()._release_inputs()
        self.jump_wait, self.was_on_ground = 0, True

    def _with_timer(self, seen):
        k = self.keys
        seen[HEAD : HEAD + KNOWN] = (float(k["sneak"]), max(0.0, 1.0 - self.t / self.limit), float(k["sprint"]), float(k["jump"]), float(k["forward"]), float(k["back"]), float(k["left"]), float(k["right"]), self.jump_wait / JUMP_WAIT)
        return seen

    # ---- Playing ----

    def _after_reset(self):
        """The reset builds courses from plain blocks. Slabs go in afterwards, and one tick lets the game show them."""
        if not self.shape_cells:
            return
        palette = sorted(set(self.shape_cells.values()))
        rows = [[x, y, z, palette.index(s)] for (x, y, z), s in self.shape_cells.items()]
        self.c.call(op="paste", palette=palette, blocks=rows, now=True)
        self.obs = self.c.step()

    def _begin_episode(self):
        self.visited = set()
        self._plan_pushes()
        self.next_push, self.pushed, self.push_airborne, self.pushes_given = 0, False, False, 0
        super()._begin_episode()
        self.closest_block = self._climb(math.hypot(self.goal_x, self.goal_z) - 1.0, self.dy)  # the edge of the start platform
        self._anchor()

    def _anchor(self):
        """Where it was when it last moved on: being stuck means standing about here, not just being no closer."""
        self.anchor, self.since_moved = list(self.obs["pos"]), 0

    def _respawn(self):
        super()._respawn()
        self._anchor()
        self.pushed = False

    def _closer(self, x, y, z):
        """Points for a block (placed or stood on) closer to the goal than any so far, by how much closer. As far as
        standing on it would leave: a block under the goal while still below it is no step closer."""
        reach = self._climb(math.hypot(self.goal_x - x, self.goal_z - z), self.dy - y + Y)
        if reach >= self.closest_block:
            return 0.0
        gained, self.closest_block = self.closest_block - reach, reach
        return self._points("closer than ever", PLACE_BONUS * gained)

    def _maybe_push(self, before, keys, clicked, turn_yaw, turn_pitch):
        """The next planned push, the first tick it fits and can be saved (clutch.rs decides, in both the game and the
        simulator). A game without the mod's push gets none."""
        plan = self.pushes
        while self.next_push < len(plan) and self.t > plan[self.next_push][1]:
            self.next_push += 1
        o = self.obs
        if self.next_push >= len(plan) or self.pushed or not o["on_ground"] or self.t < plan[self.next_push][0] or not self.can_push:
            return
        if len(o.get("solid", "")) != (2 * RADIUS + 1) ** 3:
            return
        import mcsim
        order = ("forward", "back", "left", "right", "jump", "sneak", "sprint")
        fall_line = min(Y + 1, Y + 1 + self.dy) - self.fall_margin
        try:  # the game reports feet as decimals; whatever else differs, a push is never worth stopping the round for
            v = mcsim.push_for(o["solid"], [math.floor(f) for f in o["feet"]], list(o["pos"]), list(o["vel"]), float(o["yaw"]), float(o["pitch"]), int(o["held_count"]),
                               [bool(before[k]) for k in order], [bool(keys[k]) for k in order], bool(clicked), turn_yaw, turn_pitch, [self.goal_x + 0.5, self.goal_z + 0.5],
                               fall_line, plan[self.next_push], sprinting=bool(o.get("sprinting", False)))
        except (TypeError, ValueError) as e:
            print(f"no pushes: {e}", flush=True)
            AutoEnv.can_push = False
            return
        if v is None:
            return
        try:
            self.c.call(op="push", vel=list(v))
        except RuntimeError as e:  # an older mod: no pushes in this game
            print(f"no pushes: {e}", flush=True)
            AutoEnv.can_push = False
            return
        self.next_push += 1
        self.pushes_given += 1
        self.pushed, self.push_airborne = True, False

    def _wait_dead(self):
        """It sees nothing while dead, so only people watching have to wait for it."""
        if self.realtime:
            next_tick = time.perf_counter()
            for _ in range(RESPAWN_TICKS):
                self.c.step()
                next_tick = pace(next_tick, 0.05)

    def _tick(self, move, strafe, sneak, use, jump, sprint, yaw, pitch):
        held, health = self.obs["held_count"], self.obs["health"]
        before = dict(self.keys)
        keys = self._press(dict(forward=move == 1, back=move == 2, left=strafe == 1, right=strafe == 2, sneak=bool(sneak), jump=bool(jump), sprint=bool(sprint)))
        self.since_sneak = 0 if keys["sneak"] else self.since_sneak + 1
        self.sneak_ticks += keys["sneak"]
        clicked = self._click(use) and self.since_sneak <= SNEAK_CLICK_TICKS
        turn_yaw, turn_pitch = self._aim(TURN_STEPS[yaw]), self._aim(TURN_STEPS[pitch])
        self._maybe_push(before, keys, clicked, turn_yaw, turn_pitch)
        self.obs = o = self.c.step(use=clicked, yaw=turn_yaw, pitch=turn_pitch, **keys)
        self.t += 1
        self._track_jump(o, keys["jump"])
        self._visit(o)
        reward = self._points("time", -TICK_COST)
        if use and held == 0:
            reward += self._points("click without blocks", -EMPTY_CLICK_COST)
        # Shift is free everywhere: on parkour it stops at an edge, lines up a jump, or takes a careful step.
        aims_at_block = (o.get("hit") or {}).get("type") == "block"
        if o["pitch"] < 0 and not aims_at_block:  # looking above level at nothing in reach
            reward += self._points("looking up", -LOOK_UP_COST * -o["pitch"] / 90)
        if self.pushed:
            if not o["on_ground"]:
                self.push_airborne = True
            elif self.push_airborne:
                self.pushed = False
                reward += self._points("clutched", CLUTCH_BONUS)
        if self.family == "hop":
            if self.landed < len(self.landing_order) and o["on_ground"] and self._is_on(o, self.landing_order[self.landed]):
                self.landed += 1
                reward += self._points("landed on a block", LANDING_BONUS)
        elif self.landing_order and o["on_ground"]:
            # A path is walked, so a corner can be cut past a block. Distance alone pays here.
            for i in range(self.landed, min(self.landed + 3, len(self.landing_order))):
                if self._is_on(o, self.landing_order[i]):
                    self.landed = i + 1
                    break
        now = self._distance(o)
        if o["on_ground"]:
            gained = self.ground_distance - now
            if self.deaths == 0 or gained > 0:  # walking back after a respawn pays nothing until it gets further
                reward += self._points("progress", gained)
                self.ground_distance = now
        if now < self.best - 0.01:
            self.best, self.since_best = now, 0
        else:
            self.since_best += 1
        for _ in range(max(0, held - o["held_count"])):
            self.used += 1
            reward += self._points("blocks", -(BLOCK_COST if self.used <= self.par else WASTE_COST))
        if held > o["held_count"] and o.get("clicks"):
            c = o["clicks"][-1]
            reward += self._closer(c[0], c[1], c[2])
            for i in range(3):  # a block built outside the arena must also be cleared next time
                self.dirty[i] = min(self.dirty[i], c[i])
                self.dirty[i + 3] = max(self.dirty[i + 3], c[i])
        if self.blocks_given > 0 and o["on_ground"]:
            # The block it stands on counts the same as one it placed: jumping the last gap is worth as much as
            # building it, and a block put on ground it already reached is worth nothing.
            x, y, z = o["pos"]
            reward += self._closer(math.floor(x), math.floor(y) - 1, math.floor(z))
        success = self.reached(o)
        fell = (self.fell(o) or o["health"] <= 0) and not success  # fall damage that kills is a death too, as in the game
        if success:
            for _ in range(DAMAGE_SETTLE_TICKS):  # damage from this landing arrives a moment later
                o = self.c.step()
            self.obs = o
        # Stuck: no closer for 2 s and still about where it was then. A way around something first leads away from the
        # goal, so moving on counts too, for up to 10 s without getting any closer.
        if math.dist(o["pos"], self.anchor) > MOVED_ON:
            self.anchor, self.since_moved = list(o["pos"]), 0
        else:
            self.since_moved += 1
        stalled = not (success or fell) and (self.since_best >= STALL_TICKS and self.since_moved >= STALL_TICKS or self.since_best >= DETOUR_TICKS)
        lost = max(0.0, health - o["health"])
        reward += self._points("fall damage", -(0.5 * lost))
        self.damage += lost
        clean = success and self.deaths == 0
        if success:
            reward += self._points("goal", GOAL_REWARD * (1.0 if clean else GOAL_AFTER_DEATH))
            reward += self._points("speed", SPEED_BONUS * max(0.0, 1.0 - self.t / self.limit))
            # Every block counts: one placed where it could already walk, or jumped instead of built, is worth as much
            # as one over the need. Only on courses with blocks, and never more than the need.
            if self.blocks_given > 0 and self.used < self.par:
                reward += self._points("blocks saved", SAVED_BLOCK_BONUS * (self.par - self.used))
        if fell:
            reward += self._points("death", -DEATH_COST)
        if stalled:
            reward += self._points("stuck", -STUCK_COST)
        respawned = out_of_lives = False
        if fell:
            self.deaths += 1
            out_of_lives = self.deaths >= MAX_DEATHS
            if not out_of_lives:
                self._respawn()
                respawned = True
                o = self.obs
        timed_out = self.t >= self.limit and not (success or stalled or out_of_lives)
        if (stalled or timed_out) and not fell and self._on_start_platform(o):
            reward += self._points("stayed on the start", -STAYED_HOME_COST)
        term = success or stalled or out_of_lives or timed_out
        info = {"success": success, "clean": clean, "fell": self.deaths > 0, "deaths": self.deaths, "respawned": respawned, "stalled": stalled,
                "damage": self.damage, "blocks_used": self.used, "blocks_needed": self.par, "family": self.family, "tested": self.tested,
                "landed": self.landed, "pushes": self.pushes_given, "shift_share": self.sneak_ticks / max(self.t, 1), "parts": dict(self.parts)}
        if success:
            info["blocks_per_s"] = math.hypot(self.goal_x, self.goal_z) / max(self.t / 20, 0.05)
        return float(reward) * self.reward_scale, term, False, info

