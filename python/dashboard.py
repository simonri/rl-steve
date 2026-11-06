"""Live training dashboard. Reads the files a training run writes, so it never slows the training."""
import argparse
import ast
import csv
import json
import re
import time
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
MODELS = RUNS / "models"
TEST = re.compile(r"^test at (\d+): stage (\d+) (\d+)/(\d+) blocks ([\d.]+)x(?: unused ([\d.]+))?(?: passes (\d+)/(\d+))?")
STAGE = re.compile(r"^--> stage (\d+) (\(.*\)) at (\d+) steps")
JUMPTEST = re.compile(r"^jump test at (\d+): (\d+)/(\d+)")
PASSED = re.compile(r"^all stages passed at (\d+)")
COLUMNS = ("steps", "seconds", "stage", "gap", "heights", "ticks_per_block", "success", "fell", "stalled", "damage",
           "blocks_per_s", "block_use", "unused", "episodes")


def describe(task):
    """Turns a stage's settings into something readable."""
    try:
        gap, directions, heights, yaw, ticks, start, *rest = ast.literal_eval(task)
    except (ValueError, SyntaxError):
        return task
    names = {"axis": "raka riktningar", "eight": "åtta riktningar", "any": "alla riktningar"}
    height = "platt" if heights == (0,) else f"höjd {min(heights):+d} till {max(heights):+d}"
    place = "start vid kanten" if start == "edge" else "start i mitten"
    kinds = {"island": "stödplattform", "pillar": "pelare", "wall": "vägg", "slalom": "slalom", "mixed": "blandade hinder", "jump": "hopp utan block", "treenan": "Treenan-kartan", "climb": "klättring", "hops": "enstaka block", "islands": "stanna på block", "wide_zigzag": "sicksack med hål", "run": "smal väg", "run_holes": "smal väg med hål"}
    extra = f", {kinds.get(rest[0], rest[0])}" if rest and rest[0] else ""
    phases = {"learn": "lära", "secure": "säkra", "efficient": "effektivisera"}
    phase = f", {phases.get(rest[2], rest[2])}" if len(rest) > 2 else ""
    return f"Gap {gap}, {names.get(directions, directions)}, vinkel ±{yaw}°, {height}, {ticks} ticks/block, {place}{extra}{phase}"


def newest_run():
    runs = [p for p in MODELS.glob("*/status.json")]
    return max(runs, key=lambda p: p.stat().st_mtime).parent.name


def current_timeline(rows):
    """A resume from an older model steps back in time. Only the newest pass over those steps still counts."""
    kept, before = [], float("inf")
    for row in reversed(rows):
        if row["steps"] < before:
            kept.append(row)
            before = row["steps"]
    return kept[::-1]


def read_progress(name):
    rows = []
    try:
        with open(MODELS / name / "progress.csv", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    parsed = {k: float(row[k]) for k in COLUMNS}
                    parsed["shift"] = float(row["shift"]) if row.get("shift") else None
                    rows.append(parsed)
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return current_timeline(rows)


def read_log(name):
    tests, stages, done, jumps = [], [], None, []
    try:
        lines = (RUNS / f"{name}.log").read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        m = TEST.match(line)
        if m:
            wins, games = int(m[3]), int(m[4])
            tests.append(dict(steps=int(m[1]), stage=int(m[2]), wins=wins, games=games, success=wins / max(games, 1),
                              blocks=float(m[5]), unused=float(m[6]) if m[6] else None,
                              passes=int(m[7]) if m[7] else None))
            continue
        m = STAGE.match(line)
        if m:
            stages.append(dict(stage=int(m[1]), task=describe(m[2]), steps=int(m[3])))
            continue
        m = JUMPTEST.match(line)
        if m:
            jumps.append(dict(steps=int(m[1]), success=int(m[2]) / max(int(m[3]), 1)))
            continue
        m = PASSED.match(line)
        if m:
            done = int(m[1])
    return current_timeline(tests), current_timeline(stages), done, current_timeline(jumps)


def read_levels(name):
    """Level and share of clean rounds per course kind over time, on runs with automatic difficulty."""
    rows = []
    try:
        with open(MODELS / name / "levels.csv", newline="") as f:
            for row in csv.DictReader(f):
                rows.append({k: float(v) if v else None for k, v in row.items()})
    except (OSError, ValueError):
        pass
    return current_timeline(rows)


def snapshot(name):
    run = name or newest_run()
    path = MODELS / run / "status.json"
    try:
        status = json.loads(path.read_text())
        age = time.time() - path.stat().st_mtime
    except (OSError, ValueError):
        status, age = {}, None
    tests, stages, done, jumps = read_log(run)
    return dict(run=run, status=status, age=age, progress=read_progress(run), tests=tests, stages=stages, done=done, jumps=jumps, levels=read_levels(run))


class Handler(BaseHTTPRequestHandler):
    run_name = None

    def _send(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/api/data":
            asked = parse_qs(url.query).get("run", [None])[0]
            run = asked if asked and (MODELS / asked / "status.json").exists() else self.run_name
            self._send(json.dumps(snapshot(run)).encode(), "application/json")
        elif url.path == "/":
            self._send((HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8791)
    p.add_argument("--name", help="training run; the newest one when left out")
    a = p.parse_args()
    Handler.run_name = a.name
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
