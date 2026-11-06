"""A server where whoever joins plays parkour, paths, bridges and the Treenan map, and rounds won without dying are recorded.

Every round is drawn at random, course kind and every dial, so the recordings cover all of it instead of what the player
happened to be good at. Start the game with launch.py --human, then this.
"""
import argparse
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np

import scoreboard as sb
from demo_data import trim
from auto_env import AutoEnv
from curriculum import FAMILIES
from mcclient import McClient
from skill_env import LEFTOVERS, MAX_DEATHS, Y

ROOT = Path(__file__).resolve().parent.parent
PLAYED = ("hop", "path", "bridge", "treenan")
NAMES = {"hop": "Parkour", "path": "Path", "bridge": "Bridge", "treenan": "Treenan"}
FIXED = {"ticks": 0, "faster": 0}  # plenty of time here
PASTE_ROWS = 8000  # blocks of the map sent at a time
SECONDS_PER_BLOCK = 4
MIN_SECONDS = 30
POLL = 0.05


class Round:
    """The generator and the rules from training, without the training game behind them."""

    def __init__(self, family, settings, rng, dirty):
        self.env = env = AutoEnv.__new__(AutoEnv)
        env.c, env.mix, env.dirty, env._np_random = None, 0.0, list(dirty), rng
        env.levels = None
        env._use_task((family, settings, None))
        self.spec = env._spec()
        self.family, self.settings = family, settings
        if family == "path":
            # One goal for everything: the last block of the way, not a platform after it.
            env.landing_order = env.landing_order[:-1]
            self.spec["platforms"] = self.spec["platforms"][:-1]
            env.goal_x, env.goal_z, env.dy = env.course[-1][0], env.course[-1][2], env.course[-1][1] - Y
        if family in ("bridge", "treenan"):
            self.goal = (env.goal_x, Y + env.dy, env.goal_z)  # the middle of the far platform, or a spot on the island
            self.goal_cell = (env.goal_x, Y + env.dy + 1, env.goal_z)
            self.lowest = min(Y + 1, Y + 1 + env.dy) - (env.fall_margin - 1.5)
            self.size = env.par
        else:
            self.goal = env.course[-1]
            self.goal_cell = env.landing_order[-1]
            self.lowest = min(Y + 1, min(c[1] for c in env.course) + 1)
            self.size = len(env.course)
        self.deaths, self.ticks = 0, 0
        self.limit = max(MIN_SECONDS, SECONDS_PER_BLOCK * self.size) * 20

    def update(self, o):
        """One tick. Returns "won", "died" or None. The only goal is standing on the redstone block."""
        self.ticks += 1
        if o["on_ground"] and self.env._is_on(o, self.goal_cell):
            return "won"
        if o["pos"][1] < self.lowest - 1.5:
            self.deaths += 1
            return "died"
        return None

    def meta(self):
        env = self.env
        return dict(family=self.family, settings={k: v if not isinstance(v, tuple) else list(v) for k, v in self.settings.items()}, course=env.course, poles=env.poles,
                    landing_order=env.landing_order, goal=[env.goal_x, env.goal_z], dy=env.dy, par=env.par, fall_margin=env.fall_margin, limit=self.limit, home=self.spec["pos"],
                    blocks_given=env.blocks_given, kind=env.kind, blocked=sorted(env.blocked), detour=env.detour,
                    yaw=self.spec["yaw"], pitch=self.spec["pitch"])


class Recording:
    def __init__(self, folder, number, player, meta):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"round{number:04d}.jsonl.gz"
        self.head = dict(meta=meta, player=player, started=time.time())
        self.rows = []

    def add(self, row):
        self.rows.append(row)

    def save(self, deaths, ticks, block_names):
        """Only the try that made it, without the waiting."""
        rows = trim(self.rows)
        self.head["block_names"] = block_names  # what the characters in "ids" stand for
        lines = [self.head, *rows, dict(result="won", deaths=deaths, ticks=ticks, kept=len(rows))]
        tmp = self.path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            f.write("\n".join(json.dumps(line) for line in lines) + "\n")
        tmp.replace(self.path)
        return len(rows)


class Server:
    def __init__(self, port, out):
        self.c = McClient(port)
        self.rng = np.random.default_rng()
        self.out = out
        self.force = None  # a (family, {dial: value}) to play instead of a random round, for tests
        self.number = sum(1 for _ in out.glob("*/round*.jsonl.gz")) if out.exists() else 0
        self.wins, self.games = 0, 0
        self.last_board = 0.0
        self.c.call(op="release")
        self.c.call(op="human_record", on=False)
        self.first = True
        self.dirty = [-1, Y, -1, 1, Y, 1]  # all that has to be cleared before the next course

    def draw(self):
        family = PLAYED[int(self.rng.integers(len(PLAYED)))]
        chosen = {}
        if self.force:
            family, chosen = self.force
        settings = {}
        for dial, values in FAMILIES[family].items():
            level = FIXED[dial] if dial in FIXED else int(self.rng.integers(len(values)))
            settings[dial] = chosen.get(dial, values[level])
        return Round(family, settings, self.rng, self.dirty)

    def paste_map(self, r):
        """The real map goes up first, so that the player has somewhere to arrive."""
        palette, index, rows = [], {}, []
        for (x, y, z), state in r.env.next_cells.items():
            if state not in index:
                index[state] = len(palette)
                palette.append(state)
            rows.append([x, y, z, index[state]])
        self.c.call(op="human_wipe", box=LEFTOVERS)
        for i in range(0, len(rows), PASTE_ROWS):
            self.c.call(op="human_paste", palette=palette, blocks=rows[i : i + PASTE_ROWS])

    def board(self, r, status=""):
        lines = [
            [[NAMES[r.family], "light_purple", True]],
            sb.stat("Deaths", f"{r.deaths}/{MAX_DEATHS}", sb.GOOD if r.deaths == 0 else sb.CLOSE),
            sb.stat("Time", f"{max(0, r.limit - r.ticks) // 20} s", sb.VALUE),
            sb.BLANK,
            sb.stat("Won", f"{self.wins}/{self.games}", sb.VALUE),
            sb.stat("Recorded", str(self.number), sb.DIM),
        ]
        if status:
            lines.insert(4, [[status, sb.CLOSE]])
        self.c.call(op="human_board", title=[["PARKOUR ", "white", True], ["CHALLENGE", "aqua", True]], lines=lines)

    def play_round(self, player, folder):
        r = self.draw()
        spec = r.spec
        self.c.call(op="human_record", on=False)
        if r.family == "treenan":
            self.paste_map(r)
            self.first = False
        wipe = {"wipe": LEFTOVERS} if self.first else {}
        self.first = r.family == "treenan"  # the whole map has to go before the next course
        self.c.call(op="human_build", clear=spec["clear"], platforms=spec["platforms"], goal=list(r.goal), blocks=r.env.blocks_given, pos=spec["pos"], yaw=spec["yaw"], pitch=spec["pitch"], **wipe)
        self.c.call(op="human_say", text=[[f"{NAMES[r.family]} ", "light_purple", True], ["get to the redstone block", "gray"]])
        self.c.call(op="human_ticks")
        self.c.call(op="human_record", on=True)
        rec = Recording(folder, self.number, player, r.meta())
        self.board(r)
        settled, result = False, None
        self.dirty = list(r.env.dirty)
        previous = None
        while result is None:
            time.sleep(POLL)
            reply = self.c.call(op="human_ticks")
            if "player" not in reply:
                result = "left"
                break
            for o in reply["ticks"]:
                x, y, z = o["pos"]
                if not settled:
                    # The teleport takes a moment to reach the player. Nothing before it belongs to this round.
                    settled = math.hypot(x - spec["pos"][0], z - spec["pos"][2]) < 0.5 and abs(y - spec["pos"][1]) < 1.0
                    if not settled:
                        continue
                    previous = o["pos"]
                o["vel"] = [a - b for a, b in zip(o["pos"], previous)]
                o["speed"] = math.hypot(o["vel"][0], o["vel"][2])
                previous = o["pos"]
                what = r.update(o)
                rec.add(dict(o, event=what))
                # Whatever it built stays within reach of where it has been.
                for i, v in enumerate((x, y, z)):
                    self.dirty[i] = min(self.dirty[i], int(math.floor(v)) - 6)
                    self.dirty[i + 3] = max(self.dirty[i + 3], int(math.floor(v)) + 6)
                if what == "won":
                    result = "won"
                    break
                if what == "died":
                    if r.deaths >= MAX_DEATHS:
                        result = "died"
                        break
                    self.c.call(op="human_move", pos=spec["pos"], yaw=spec["yaw"], pitch=spec["pitch"])
                    self.c.call(op="human_say", text=[["One more try", "yellow"]])
                    settled = False
                    break
                if r.ticks >= r.limit:
                    result = "time"
                    break
            if time.time() - self.last_board >= 1.0:
                self.last_board = time.time()
                self.board(r)
        self.c.call(op="human_record", on=False)
        self.finish_round(r, rec, result)

    def finish_round(self, r, rec, result):
        kept = 0
        if result == "won" and r.deaths == 0:  # only rounds won without dying are worth copying
            kept = rec.save(r.deaths, r.ticks, self.c.call(op="human_ticks").get("block_names", {}))
            self.number += 1
        print(f"{r.family}: {result}, deaths {r.deaths}, {r.ticks} ticks, kept {kept}", flush=True)
        if result == "left":
            return
        self.games += 1
        self.wins += result == "won"

    def serve(self):
        while True:
            reply = self.c.call(op="human_ticks")
            player = reply.get("player")
            if not player:
                self.first = True
                time.sleep(0.5)
                continue
            folder = self.out / time.strftime(f"%Y%m%d-%H%M-{player}")
            print("playing:", player, flush=True)
            while True:
                self.play_round(player, folder)
                if "player" not in self.c.call(op="human_ticks"):
                    print("left:", player, flush=True)
                    break
                time.sleep(1.0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=25608)
    p.add_argument("--out", default=str(ROOT / "runs" / "demos"))
    a = p.parse_args()
    Server(a.port, Path(a.out)).serve()
