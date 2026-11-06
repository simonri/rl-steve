"""Bedwars maps made up fresh every round: from the own base, across one to three gaps of void, to a spot on another island.

Islands sit at different heights and in different directions, with walls, pillars, trees, beds and bed defences
on them, the way real maps look. Every round can be done: the goal is a flat 3 by 3 with room to stand, nothing walls in
the start or the goal, the blocks handed out cover every gap and every climb on the cheapest way, and no island lies
below the line where a fall counts as dying.

    layout(task, rng, base_y) -> Layout
"""
import heapq
import math
from dataclasses import dataclass, field

TEAMS = ("red", "blue", "lime", "yellow", "light_blue", "white", "pink", "gray")
GROUND = ("minecraft:{team}_wool", "minecraft:{team}_terracotta", "minecraft:stone_bricks", "minecraft:smooth_stone", "minecraft:oak_planks")
MID_GROUND = ("minecraft:grass_block[snowy=false]", "minecraft:stone_bricks", "minecraft:andesite", "minecraft:moss_block")
UNDER = ("minecraft:stone", "minecraft:dirt", "minecraft:cobblestone", "minecraft:andesite")
LOG = "minecraft:oak_log[axis=y]"
LEAVES = "minecraft:oak_leaves[persistent=true]"  # only properties every version has: a state it cannot read is left out of the paste
END_STONE = "minecraft:end_stone"
DEEPEST = 3  # no island top lies more than this below the lower of start and goal: a fall counts from 6 below
HIGHEST = 10  # nor more than this above the start
REACH = 90  # every block within this of the start, inside what the game has loaded and what a wipe clears
CLUTTER = {"none": (0, 0), "some": (1, 2), "many": (3, 5)}  # obstacles per island


@dataclass
class Island:
    cx: int
    cz: int
    y: int  # height of the top layer
    hx: int  # half sizes: the top covers cx - hx .. cx + hx
    hz: int
    kind: str  # "base", "step", "diamond", "mid", "enemy"

    def inside(self, x, z, margin=0):
        return abs(x - self.cx) <= self.hx - margin and abs(z - self.cz) <= self.hz - margin

    def edge(self, ux, uz):
        """Distance from the centre to the edge of the top along the unit direction (ux, uz)."""
        return min(self.hx / max(abs(ux), 1e-9), self.hz / max(abs(uz), 1e-9))

    def near(self, other, room):
        return abs(self.cx - other.cx) <= self.hx + other.hx + room and abs(self.cz - other.cz) <= self.hz + other.hz + room


@dataclass
class Layout:
    cells: dict  # (x, y, z) -> block state, already moved so the start is at (0, base_y, 0)
    goal: tuple  # (x, y, z) of the ground block under the middle of the goal
    blocked: set  # columns with something at walking height that the way should go around
    par: int  # blocks for the cheapest way: every void column crossed, and two per block climbed
    islands: list = field(default_factory=list)


def _ground_cells(isl, top, under, cells, rng):
    """The top layer, and two narrower layers below it so it looks like an island and not a plate."""
    for x in range(isl.cx - isl.hx, isl.cx + isl.hx + 1):
        for z in range(isl.cz - isl.hz, isl.cz + isl.hz + 1):
            cells[(x, isl.y, z)] = top
            for d in (1, 2):
                if isl.inside(x, z, d) and not (d == 2 and rng.random() < 0.3):
                    cells[(x, isl.y - d, z)] = under


def _free_spots(isl, cells, keep_out):
    """Tops where a 3 by 3 is flat ground with two empty blocks above it."""
    spots = []
    for x in range(isl.cx - isl.hx + 1, isl.cx + isl.hx):
        for z in range(isl.cz - isl.hz + 1, isl.cz + isl.hz):
            ok = (x, z) not in keep_out
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    c = (x + dx, isl.y, z + dz)
                    if c not in cells or (x + dx, isl.y + 1, z + dz) in cells or (x + dx, isl.y + 2, z + dz) in cells:
                        ok = False
            if ok:
                spots.append((x, isl.y, z))
    return spots


def _bed(isl, x, z, facing, team, cells):
    """A bed two blocks long: the foot at (x, z), the head one further toward `facing`."""
    step = {"east": (1, 0), "west": (-1, 0), "south": (0, 1), "north": (0, -1)}[facing]
    cells[(x, isl.y + 1, z)] = f"minecraft:{team}_bed[facing={facing},occupied=false,part=foot]"
    cells[(x + step[0], isl.y + 1, z + step[1])] = f"minecraft:{team}_bed[facing={facing},occupied=false,part=head]"
    return [(x, z), (x + step[0], z + step[1])]


def _defence(isl, bed, team, cells):
    """Wool around and over the bed, end stone around that: what a base looks like a few minutes in."""
    inner, outer = set(), set()
    for bx, bz in bed:
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                inner.add((bx + dx, bz + dz))
                for ex in (-2, -1, 0, 1, 2):
                    for ez in (-2, -1, 0, 1, 2):
                        outer.add((bx + ex, bz + ez))
    columns = []
    for x, z in sorted(outer):
        if not isl.inside(x, z):
            continue
        if (x, z) in bed:
            cells[(x, isl.y + 2, z)] = f"minecraft:{team}_wool"
        elif (x, z) in inner:
            cells[(x, isl.y + 1, z)] = f"minecraft:{team}_wool"
            cells[(x, isl.y + 2, z)] = f"minecraft:{team}_wool"
            columns.append((x, z))
        elif max(abs(x - bx) for bx, _ in bed) <= 2:
            cells[(x, isl.y + 1, z)] = END_STONE
            columns.append((x, z))
    return columns


def _obstacle(isl, rng, cells, keep_out):
    """One wall, pillar or tree on the island top, never on or next to a spot that has to stay free.
    Returns the columns it stands in at walking height, or None when it did not fit."""
    kind = ("wall", "pillar", "tree")[int(rng.integers(3))]  # fences left out for now
    for _ in range(12):
        x = int(rng.integers(isl.cx - isl.hx, isl.cx + isl.hx + 1))
        z = int(rng.integers(isl.cz - isl.hz, isl.cz + isl.hz + 1))
        if kind in ("wall", "fence"):
            along = (1, 0) if rng.random() < 0.5 else (0, 1)
            n = int(rng.integers(3, 7))
            cols = [(x + along[0] * i, z + along[1] * i) for i in range(n)]
        else:
            cols = [(x, z)]
        if any(not isl.inside(cx, cz) or (cx, cz) in keep_out or (cx, isl.y + 1, cz) in cells for cx, cz in cols):
            continue
        if kind == "wall":
            height = int(rng.integers(2, 4))
            state = f"minecraft:{TEAMS[int(rng.integers(len(TEAMS)))]}_wool" if rng.random() < 0.5 else "minecraft:stone_bricks"
            for cx, cz in cols:
                for h in range(1, height + 1):
                    cells[(cx, isl.y + h, cz)] = state
        elif kind == "fence":
            ax = cols[1][0] != cols[0][0]
            for i, (cx, cz) in enumerate(cols):
                a, b = (i > 0, i < len(cols) - 1)
                east, west, south, north = (b, a, False, False) if ax else (False, False, b, a)
                cells[(cx, isl.y + 1, cz)] = f"minecraft:oak_fence[east={str(east).lower()},north={str(north).lower()},south={str(south).lower()},waterlogged=false,west={str(west).lower()}]"
        elif kind == "pillar":
            for h in range(1, int(rng.integers(3, 6)) + 1):
                cells[(x, isl.y + h, z)] = "minecraft:stone_bricks"
        else:
            trunk = int(rng.integers(3, 5))
            for h in range(1, trunk + 1):
                cells[(x, isl.y + h, z)] = LOG
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for h in (trunk, trunk + 1):
                        c = (x + dx, isl.y + h, z + dz)
                        if c not in cells:
                            cells[c] = LEAVES
            cells.setdefault((x, isl.y + trunk + 2, z), LEAVES)
        return cols
    return None


def _par(islands, cells, start, goal):
    """Blocks for the cheapest way from start to goal: one per void column crossed (sideways steps only, never a
    corner), and two per block climbed from one island to the next higher one."""
    top = {}
    for isl in islands:
        for x in range(isl.cx - isl.hx, isl.cx + isl.hx + 1):
            for z in range(isl.cz - isl.hz, isl.cz + isl.hz + 1):
                top[(x, z)] = isl.y
    lo_x = min(i.cx - i.hx for i in islands) - 2
    hi_x = max(i.cx + i.hx for i in islands) + 2
    lo_z = min(i.cz - i.hz for i in islands) - 2
    hi_z = max(i.cz + i.hz for i in islands) + 2
    # State: column and the height of the last ground stood on, since climbing is paid for at the next island.
    best = {(start[0], start[2], start[1]): 0}
    queue = [(0, start[0], start[2], start[1])]
    while queue:
        cost, x, z, h = heapq.heappop(queue)
        if (x, z) == (goal[0], goal[2]):
            return cost + 2 * max(goal[1] - h, 0)
        if best.get((x, z, h), math.inf) < cost:
            continue
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, nz = x + dx, z + dz
            if not (lo_x <= nx <= hi_x and lo_z <= nz <= hi_z):
                continue
            if (nx, nz) in top:
                nh = top[(nx, nz)]
                step = 2 * max(nh - h, 0)
            else:
                nh, step = h, 1  # a bridge block at the height it came from
            c = cost + step
            if c < best.get((nx, nz, nh), math.inf):
                best[(nx, nz, nh)] = c
                heapq.heappush(queue, (c, nx, nz, nh))
    raise RuntimeError("goal not reachable")


def layout(task, rng, base_y, most_blocks=64):
    """One map. Raises RuntimeError when it found nothing that fits; the caller tries again."""
    legs = task["legs"]
    clutter = CLUTTER[task["clutter"]]
    target = task["target"]
    team, enemy = (TEAMS[i] for i in rng.choice(len(TEAMS), 2, replace=False))
    budget = task.get("budget", (2.0, 2))
    for attempt in range(80):
        # A map that needs more blocks than one stack gets shorter gaps and smaller climbs on the next try.
        gap_most = max(3, task["gap"] - attempt // 4)
        height = max(0, task["height"] - attempt // 8)
        angle = float(rng.uniform(0, 2 * math.pi)) if task["turns"] == "any" else math.pi / 2 * int(rng.integers(4))
        base = Island(0, 0, base_y, int(rng.integers(3, 6)), int(rng.integers(3, 6)), "base")
        islands = [base]
        ok = True
        for leg in range(legs):
            last = leg == legs - 1
            if leg:
                angle += float(rng.uniform(-math.pi / 4, math.pi / 4)) if task["turns"] == "any" else math.pi / 2 * int(rng.choice((-1, 0, 0, 1)))
            ux, uz = math.cos(angle), math.sin(angle)
            if not last:
                hx, hz, kind = int(rng.integers(1, 3)), int(rng.integers(1, 3)), "step"
            elif target == "diamond":
                hx = hz = 3
                kind = "diamond"
            elif target == "mid":
                hx, hz, kind = int(rng.integers(5, 8)), int(rng.integers(5, 8)), "mid"
            else:
                hx, hz, kind = int(rng.integers(4, 6)), int(rng.integers(4, 6)), "enemy"
            prev = islands[-1]
            gap = int(rng.integers(max(3, gap_most // 2), gap_most + 1))
            new = Island(0, 0, 0, hx, hz, kind)
            reach = prev.edge(ux, uz) + gap + 1 + new.edge(ux, uz)
            new.cx, new.cz = int(round(prev.cx + ux * reach)), int(round(prev.cz + uz * reach))
            new.y = prev.y + (int(rng.integers(-height, height + 1)) if height else 0)
            new.y = min(max(new.y, base_y - DEEPEST), base_y + HIGHEST)
            if any(new.near(o, 2) for o in islands) or math.hypot(new.cx, new.cz) + max(hx, hz) > REACH:
                ok = False
                break
            islands.append(new)
        if not ok:
            continue
        low = min(base_y, islands[-1].y)
        if any(i.y < low - DEEPEST for i in islands):
            continue  # an island this low would count as a fall
        cells, blocked = {}, set()
        team_ground = GROUND[int(rng.integers(len(GROUND)))].format(team=team)
        for isl in islands:
            if isl.kind == "mid":
                top = MID_GROUND[int(rng.integers(len(MID_GROUND)))]
            elif isl.kind == "diamond":
                top = "minecraft:smooth_stone"
            elif isl.kind == "enemy":
                top = GROUND[int(rng.integers(len(GROUND)))].format(team=enemy)
            else:
                top = team_ground
            _ground_cells(isl, top, UNDER[int(rng.integers(len(UNDER)))], cells, rng)
        # The own bed and the start: by the bed (spawn) or anywhere flat on the base.
        facing = ("east", "west", "south", "north")[int(rng.integers(4))]
        bx, bz = base.cx + int(rng.integers(-1, 2)), base.cz + int(rng.integers(-1, 2))
        bed = _bed(base, bx, bz, facing, team, cells)
        keep = {(x + dx, z + dz) for x, z in bed for dx in (-1, 0, 1) for dz in (-1, 0, 1)}
        spots = [s for s in _free_spots(base, cells, keep)]
        if task["start"] == "spawn":
            spots = sorted(spots, key=lambda s: math.hypot(s[0] - bx, s[2] - bz))[:4]
        if not spots:
            continue
        start = spots[int(rng.integers(len(spots)))]
        # The goal's island: what makes it that kind of island, then the goal spot.
        goal_isl = islands[-1]
        if goal_isl.kind == "diamond":
            cells[(goal_isl.cx, goal_isl.y + 1, goal_isl.cz)] = "minecraft:diamond_block"
            blocked.add((goal_isl.cx, goal_isl.cz))
        elif goal_isl.kind == "enemy":
            ebed = _bed(goal_isl, goal_isl.cx, goal_isl.cz, ("east", "west", "south", "north")[int(rng.integers(4))], enemy, cells)
            blocked.update(_defence(goal_isl, ebed, enemy, cells))
        elif goal_isl.kind == "mid" and rng.random() < 0.5:
            # A raised middle, as on most maps: the goal can be up on it, and getting there means building up.
            rise = int(rng.integers(1, 4))
            px, pz = max(2, goal_isl.hx - 3), max(2, goal_isl.hz - 3)
            for x in range(goal_isl.cx - px, goal_isl.cx + px + 1):
                for z in range(goal_isl.cz - pz, goal_isl.cz + pz + 1):
                    for h in range(1, rise + 1):
                        cells[(x, goal_isl.y + h, z)] = cells[(x, goal_isl.y, z)]
            plateau = Island(goal_isl.cx, goal_isl.cz, goal_isl.y + rise, px, pz, "mid")
            islands.append(plateau)
        goal_spots = _free_spots(goal_isl, cells, set())
        if islands[-1] is not goal_isl and (rng.random() < 0.6 or not goal_spots):
            goal_spots = _free_spots(islands[-1], cells, set())  # up on the raised middle
        if not goal_spots:
            continue
        goal = goal_spots[int(rng.integers(len(goal_spots)))]
        gkeep = {(goal[0] + dx, goal[2] + dz) for dx in range(-2, 3) for dz in range(-2, 3)}
        skeep = {(start[0] + dx, start[2] + dz) for dx in range(-2, 3) for dz in range(-2, 3)}
        # Things in the way, never on the start, the goal or the beds.
        lo, hi = clutter
        for isl in islands[: legs + 1]:
            for _ in range(int(rng.integers(lo, hi + 1)) if hi else 0):
                cols = _obstacle(isl, rng, cells, keep | gkeep | skeep)
                if cols:
                    blocked.update(cols)
        if any((goal[0] + dx, goal[1] + h, goal[2] + dz) in cells for dx in (-1, 0, 1) for dz in (-1, 0, 1) for h in (1, 2)):
            continue
        par = _par(islands, cells, start, goal)
        if par * budget[0] + budget[1] > most_blocks:
            continue  # more blocks than it can hold: a shorter map instead
        # Everything moved so the start is at (0, base_y, 0), as on every other course.
        mx, my, mz = -start[0], base_y - start[1], -start[2]
        moved = {(x + mx, y + my, z + mz): s for (x, y, z), s in cells.items()}
        if max(math.hypot(x, z) for x, _, z in moved) > REACH:
            continue
        return Layout(moved, (goal[0] + mx, goal[1] + my, goal[2] + mz), {(x + mx, z + mz) for x, z in blocked}, par, islands)
    raise RuntimeError("no bedwars map fits")
