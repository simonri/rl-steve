"""The Treenan Bedwars map as training tasks: from a spot beside the bed to a flat grass spot on another island."""
import json
import math
from pathlib import Path

import map_reader

ROOT = Path(__file__).resolve().parent.parent
MAP = "/home/simon/Downloads/hypixel-bedwars-treenan-map-e950/Hypixel Bedwars Treenan Map"
CACHE = ROOT / "runs" / "maps" / "treenan.json"
WEST, CENTER, SIDE = (-86, -60, -14, 12), (-21, 22, -22, 20), (-41, -23, 25, 43)
BED = (-66.5, 0.5)
# Island edges facing each other, from reading the map: (x, z).
EDGE_STRAIGHT, EDGE_DIAGONAL, EDGE_CENTER, EDGE_SIDE = (-60, -1), (-65, 6), (-21, -1), (-38, 29)
RENAMED = {"minecraft:grass": "minecraft:short_grass"}
GRASS = "minecraft:grass_block"
MAX_DROP, MAX_CLIMB = 8, 3
MIN_GAP, MAX_GAP = 10, 30


def load_rows():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for x, y, z, name, props in map_reader.blocks(MAP):
            state = RENAMED.get(name, name)
            if props:
                state += "[" + ",".join(f"{k}={v}" for k, v in sorted(props.items())) + "]"
            rows.append((x, y, z, state))
        CACHE.write_text(json.dumps(rows))
    return [tuple(r) for r in json.loads(CACHE.read_text())]


def in_box(x, z, box):
    return box[0] <= x <= box[1] and box[2] <= z <= box[3]


NOT_A_FLOOR = ("slab", "stairs", "fence", "wall", "pane", "_bed", "carpet", "door", "sign", "banner", "torch", "button", "pressure_plate",
               "flower", "grass[", "short_grass", "tall_grass", "fern", "sapling", "water", "lava", "chest", "ladder", "vine", "rail", "lantern",
               "pot", "skull", "head", "snow", "bars", "poppy", "dandelion", "air")


def is_floor(state, any_top):
    """Grass, or with any_top every whole block that can be stood on flat."""
    if not any_top:
        return state.startswith(GRASS)
    name = state.removeprefix("minecraft:")
    return not any(k in name for k in NOT_A_FLOOR) or name.startswith("grass_block")


def flat_spots(cells, box, any_top=False):
    """Centres of 3x3 floors at one height (grass, or any whole block), with nothing to bump into in the two blocks above."""
    spots = []
    for (x, y, z), state in cells.items():
        if not in_box(x, z, box) or not is_floor(state, any_top):
            continue
        ok = True
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                ground = cells.get((x + dx, y, z + dz), "")
                if not is_floor(ground, any_top) or (x + dx, y + 1, z + dz) in cells or (x + dx, y + 2, z + dz) in cells:
                    ok = False
        if ok:
            spots.append((x, y, z))
    return sorted(spots)


class Treenan:
    def __init__(self):
        self.cells = {(x, y, z): state for x, y, z, state in load_rows()}
        self.boxes = {"start": WEST, "center": CENTER, "side": SIDE}
        self.parts = {name: {c: s for c, s in self.cells.items() if in_box(c[0], c[2], box)} for name, box in self.boxes.items()}
        home = flat_spots(self.cells, WEST)
        if not home:
            raise RuntimeError("no flat grass near the bed")
        self.start = min(home, key=lambda s: math.hypot(s[0] + 0.5 - BED[0], s[2] + 0.5 - BED[1]))
        self.homes = home  # every flat grass spot on the bed island, for starting anywhere on it
        self.spots, self.any_spots = {}, {}
        for name, box in (("center", CENTER), ("side", SIDE)):
            drop = self.start[1] - MAX_DROP
            climb = self.start[1] + MAX_CLIMB
            self.spots[name] = [s for s in flat_spots(self.cells, box) if drop <= s[1] <= climb]
            self.any_spots[name] = [s for s in flat_spots(self.cells, box, any_top=True) if drop <= s[1] <= climb]

    def layout(self, target, gap, rng, base_y, rise=0, start=None, any_top=False):
        """Moves the bed island so the start spot is at (0, base_y, 0), and the other island so the gap between
        the edges facing each other is the given number of blocks. Heights are as on the map, with the other island
        `rise` blocks higher (or lower when negative). `start` is a spot on the bed island, by default beside the bed."""
        sx, sy, sz = start or self.start
        move = (-sx, base_y - sy, -sz)
        edge, goal_edge, name = (EDGE_STRAIGHT, EDGE_CENTER, "center") if target == "straight" else (EDGE_DIAGONAL, EDGE_SIDE, "side")
        start_edge = (edge[0] + move[0], edge[1] + move[2])
        if target == "straight":
            near = (start_edge[0] + gap + 1, start_edge[1])
        else:
            step = round(gap / math.sqrt(2))
            near = (start_edge[0] + step + 1, start_edge[1] + step + 1)
        other = (near[0] - goal_edge[0], base_y - sy + rise, near[1] - goal_edge[1])
        placed = {(c[0] + move[0], c[1] + move[1], c[2] + move[2]): s for c, s in self.parts["start"].items()}
        self.last_other = set()  # the cells of the other island, for telling the two apart
        for c, s in self.parts[name].items():
            moved = (c[0] + other[0], c[1] + other[1], c[2] + other[2])
            placed[moved] = s
            self.last_other.add(moved)
        choices = self.any_spots[name] if any_top else self.spots[name]
        spot = choices[int(rng.integers(len(choices)))]
        goal = (spot[0] + other[0], spot[1] + other[1], spot[2] + other[2])
        return placed, goal
