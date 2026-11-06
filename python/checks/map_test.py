"""Test server on the real Treenan map. A watcher selects item 1 (straight) or 2 (diagonal). The whole area is
cleared and two islands are built with a random gap of 10 to 30 blocks. Nobody plays yet, this is for looking."""
import _path  # noqa: F401  (python/ on the import path)
import argparse
import math
import random
import time

import scoreboard as sb
from skill_env import SkillEnv, Y
from treenan import CENTER, SIDE, WEST, load_rows

# Where each island's edge is, found by reading the map: (x, z, height of the ground there).
EDGES = {"west_straight": (-60, -1, 66), "west_diagonal": (-65, 6, 66), "center": (-21, -1, 62), "side": (-38, 29, 67)}
WIPE = [-90, 60, -90, 130, 190, 130]  # everything a task can reach, cleared whenever a new one is made


def island(rows, box):
    return [r for r in rows if box[0] <= r[0] <= box[1] and box[2] <= r[2] <= box[3]]


def plan_task(rows, kind, length):
    """World blocks for the two islands, with the gap between their edges set to the given length."""
    start_edge = EDGES["west_straight" if kind == "straight" else "west_diagonal"]
    goal_edge, goal_box = (EDGES["center"], CENTER) if kind == "straight" else (EDGES["side"], SIDE)
    if kind == "straight":
        start_at, near, goal = (1, 0), (2 + length, 0), (2 + length + 2, 0)
    else:
        step = length / math.sqrt(2)
        far = 2 + round(step)
        start_at, near, goal = (1, 1), (far, far), (far + 2, far + 2)
    placed = {}
    # Both ground levels end up at the same height, and the top of the start ground at Y.
    moves = ((WEST, start_edge, start_at), (goal_box, goal_edge, near))
    for box, edge, at in moves:
        dx, dy, dz = at[0] - edge[0], Y - edge[2], at[1] - edge[1]
        for x, y, z, state in island(rows, box):
            placed[(x + dx, y + dy, z + dz)] = state
    return placed, goal


class Server:
    def __init__(self, port):
        self.env = SkillEnv(port, stage=0)
        self.c = self.env.c
        self.rows = load_rows()
        self.result = "Select item 1 or 2"
        self.built = 0

    def tick(self):
        self.c.step()

    def paste(self, cells):
        """Sends blocks in pieces, with one game tick after each so the server applies them."""
        palette, index, rows = [], {}, []
        for (x, y, z), state in cells.items():
            if state not in index:
                index[state] = len(palette)
                palette.append(state)
            rows.append([x, y, z, index[state]])
        for i in range(0, len(rows), 8000):
            self.c.call(op="paste", palette=palette, blocks=rows[i : i + 8000])
            self.tick()

    def build(self, kind, length=None):
        """Clears everything in the area, then builds the two islands and shows them to whoever is watching."""
        length = length or random.randint(10, 30)
        self.c.call(op="wipe", box=WIPE)
        self.tick()
        placed, goal = plan_task(self.rows, kind, length)
        self.paste(placed)
        self.c.call(op="watch_at", camera=view(goal))
        self.tick()
        self.built += 1
        self.result = f"{kind} gap {length}"
        print(self.result, f"{len(placed)} blocks", flush=True)
        return placed, length

    def idle_board(self):
        self.env.sidebar([["TREENAN TEST", "aqua", True]], [
            [["1", "green", True], [" Straight (emerald)", "white"]],
            [["2", "aqua", True], [" Diagonal (diamond)", "white"]],
            sb.BLANK,
            [[self.result[:30], "yellow"]],
            sb.stat("Built", str(self.built), sb.VALUE),
        ])
        self.tick()
        self.c.call(op="release")

    def serve(self):
        self.build("straight")
        self.idle_board()
        while True:
            wanted = self.c.call(op="menu").get("request")
            if wanted:
                self.build(wanted)
                self.idle_board()
            time.sleep(0.2)


def view(goal):
    """A spot to the side of the gap, high enough to see both islands. Yaw and pitch face the middle."""
    gx, gz = goal
    mx, mz = gx / 2, gz / 2
    length = max(math.hypot(gx, gz), 1.0)
    away, up = length * 0.7 + 22, length * 0.3 + 14
    x, z = mx + gz / length * away, mz - gx / length * away
    yaw = math.degrees(math.atan2(-(mx - x), mz - z))
    pitch = math.degrees(math.atan2(up, away))
    return [x, Y + 1 + up, z, yaw, pitch]


def solid_outside(server, placed):
    """Reads the whole wiped area back from the game and lists everything that is not part of the plan."""
    wrong = []
    x0, y0, z0, x1, y1, z1 = WIPE
    for x in range(x0, x1 + 1):
        names = server.c.call(op="blocks", min=[x, y0, z0], max=[x, y1, z1])["names"]
        i = 0
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                if names[i] not in ("minecraft:air", "minecraft:void_air", "minecraft:cave_air") and (x, y, z) not in placed:
                    wrong.append((x, y, z, names[i]))
                i += 1
    return wrong


def self_test(server):
    """Builds tasks of both kinds and checks that nothing is left of the one before."""
    for kind, length in (("straight", 30), ("diagonal", 10), ("diagonal", 30), ("straight", 10)):
        placed, _ = server.build(kind, length)
        # The check reads the bot's own client, which gets the changes as packets over a few ticks.
        for _ in range(80):
            server.tick()
            time.sleep(0.02)
        wrong = solid_outside(server, placed)
        sample = random.Random(1).sample(sorted(placed), 400)
        bad = 0
        for cell in sample:
            name = server.c.call(op="blocks", min=list(cell), max=list(cell))["names"][0]
            bad += name != placed[cell].split("[")[0]
        print(f"{kind} gap {length}: left over from before {len(wrong)}, wrong map blocks {bad}/400", flush=True)
        assert not wrong, wrong[:5]
        assert bad == 0
    beds = [c for c, s in plan_task(server.rows, "straight", 10)[0].items() if "_bed" in s]
    print("beds on the start island:", len(beds), flush=True)
    assert beds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=25604)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    server = Server(a.port)
    if a.self_test:
        self_test(server)
        server.c.close()
    else:
        server.serve()
