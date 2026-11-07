"""How strong the model is at fighting, measured against frozen references it never trains against.

Why: a win rate against the opponents it trains with proves nothing, it can rise while the model goes round in
circles between styles (docs/training-design.md#referenser). So every so often the model fights a fixed set of
references in a few fixed kinds of fight, and every result is kept.

The references: the scripted bots, and frozen copies of the model in runs/models/<run>/references/. The first copy
is the anchor, Elo 1000. Every rating adds its games to references/games.jsonl, and all ratings, the model's at
every rating and every reference's, are fitted together from all games ever played (a Bradley-Terry fit, Elo's own
scale: 400 points is ten to one odds). Before, each new reference got the Elo of the one rating it was added at and
kept it, so the errors added up link by link, and with 32 rounds in one kind of fight auto7's Elo swung 300 points
from one rating to the next (1702, 1347, 1590). Now a rating is 64 rounds in each of three kinds of fight per
reference, and comes with its 95 % interval (docs/training-design.md#mata).
"""
import json
import math
import os
from pathlib import Path

import numpy as np

import curriculum
from benchmark import first_rounds
from opponents import Opponents
from sim_env import SimVecEnv

BOT_REFERENCES = ("slow walker", "medium walker", "walker", "strafer", "combo", "pusher")
# The kinds of fight every reference is played in: a wide platform with wood swords; an island with void all round,
# iron and knockback; a bridge between islands with stone swords. One kind alone measured one narrow skill.
FIGHTS = {
    "platform": dict(arena=0, distance=1, yaw=1, gear=0, kb=0, count=0),
    "island": dict(arena=1, distance=1, yaw=1, gear=2, kb=1, count=0),
    "bridge": dict(arena=2, distance=2, yaw=1, gear=1, kb=0, count=0),
}
ROUNDS = 64  # per reference and kind of fight
ANCHOR_ELO = 1000.0
PRIOR_SD = 800.0  # a weak pull of every rating toward 1000: a clean sweep says "much better", not how much
# When the model scores this much against the newest frozen copy of itself (the strongest reference), a new frozen copy
# joins the references. 0.75 is about 190 Elo above it. Only the copies count: the bots, beaten ~95 %, carried an
# average over all references past the line against an equal copy, and a near-copy joined at every rating.
ADD_AT = 0.75
MAX_SNAPSHOTS = 4  # the anchor and the three newest: each costs ROUNDS fights per kind a rating
# A bot the model scored at least SURE against (or at most 1 - SURE) last time is skipped, except every RECHECK_EVERY-th
# rating, which plays them all and would catch forgetting. Why: a 98 % result says almost nothing about the rating
# (OpenAI Five only played references within 15-85 %, arXiv 1912.06680 appendix J), and the bots were 6 of 10
# references. The frozen copies are always played: outgrown and the pool use them.
SURE = 0.95
RECHECK_EVERY = 4


def elo_from_score(score, against):
    """The rating that scores `score` (wins + half the draws, per round) against a player rated `against`."""
    s = min(max(score, 0.02), 0.98)  # a clean sweep says only "much better", not how much
    return against + 400.0 * math.log10(s / (1.0 - s))


def fit(games, anchor, fixed=ANCHOR_ELO):
    """Ratings for everyone in `games` [(a, b, points of a, rounds)] together (Bradley-Terry by Newton's method, with a
    weak prior): {name: (elo, half-width of its 95 % interval)}. `anchor` is held at `fixed`."""
    names = sorted({g[0] for g in games} | {g[1] for g in games} | {anchor})
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    r = np.zeros(k)  # in natural units: elo = fixed + r * 400 / ln 10
    scale = 400.0 / math.log(10.0)
    prior = (scale / PRIOR_SD) ** 2
    free = [i for i in range(k) if names[i] != anchor]
    for _ in range(100):
        g = -prior * r
        h = -prior * np.eye(k)
        for a, b, pts, n in games:
            i, j = idx[a], idx[b]
            p = 1.0 / (1.0 + math.exp(r[j] - r[i]))
            d = pts - n * p
            g[i] += d
            g[j] -= d
            w = n * p * (1 - p)
            h[i, i] -= w
            h[j, j] -= w
            h[i, j] += w
            h[j, i] += w
        step = np.zeros(k)
        step[free] = np.linalg.solve(h[np.ix_(free, free)], -g[free])
        r += step
        if np.abs(step).max() < 1e-7:
            break
    cov = np.zeros((k, k))
    cov[np.ix_(free, free)] = np.linalg.inv(-h[np.ix_(free, free)])
    return {n: (fixed + scale * r[idx[n]], 1.96 * scale * math.sqrt(max(cov[idx[n], idx[n]], 0.0))) for n in names}


def fit_elo(results):
    """Kept for callers that have scores against fixed ratings [(score, rating, rounds)]: the one rating that best
    explains them (the same fit, the references held where they are)."""
    def loglik(r):
        total = 0.0
        for score, rating, n in results:
            score = _clamped(score, n)
            e = min(max(1.0 / (1.0 + 10.0 ** ((rating - r) / 400.0)), 1e-9), 1 - 1e-9)
            total += n * (score * math.log(e) + (1.0 - score) * math.log(1.0 - e))
        return total

    grid = np.arange(-1000.0, 4000.0, 5.0)
    best = grid[int(np.argmax([loglik(r) for r in grid]))]
    fine = np.arange(best - 5.0, best + 5.0, 0.25)
    return float(fine[int(np.argmax([loglik(r) for r in fine]))])


def _clamped(score, n):
    """A score as `n` rounds can tell it: a clean sweep counts as half a round short of one."""
    return min(max(score, 0.5 / n), 1.0 - 0.5 / n)


def outgrown(scores, references, fresh=()):
    """The model is well past the newest frozen copy of itself (ADD_AT), so a new copy should join the references. The
    bots do not count, nor a copy made in this very rating."""
    older = [r for r in references if r["kind"] == "snapshot" and r["name"] not in fresh]
    return bool(older) and scores.get(older[-1]["name"], 0.0) >= ADD_AT


class Rating:
    """For train_sim: rates the model against the references, adding the anchor the first time."""

    def __init__(self, run_folder, n=ROUNDS):
        self.folder = Path(run_folder) / "references"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index = self.folder / "references.json"
        self.games_file = self.folder / "games.jsonl"
        self.references = json.loads(self.index.read_text()) if self.index.exists() else []
        # A reference whose file is gone (a stop at the wrong moment) is dropped rather than crash the next rating.
        self.references = [r for r in self.references if r["kind"] == "bot" or (self.folder / r["file"]).exists()]
        self.vec = SimVecEnv(n, curriculum.Curriculum().levels, seed=424242)  # its own rounds: training's are untouched
        self.policies = {}

    def _policy(self, ref, device):
        if ref["name"] not in self.policies:
            import train_auto as ta  # here, not at the top: train_auto is the trainer that imports this module
            self.policies[ref["name"]] = ta.load_policy(self.folder / ref["file"], device)
        return self.policies[ref["name"]]

    def _play(self, policy, ref, fight, device):
        """Points (wins + half the draws) and rounds of `policy` against `ref` in one kind of fight."""
        levels = curriculum.Curriculum().levels
        levels["duel"].update(FIGHTS[fight], bot=curriculum.BOTS.index(ref["name"]) if ref["kind"] == "bot" else 0)
        self.vec.env_method("set_levels", levels)
        self.vec.env_method("set_exam", "duel", True)
        opp = None
        if ref["kind"] != "bot":
            opp = Opponents(self.vec.num_envs)
            opp.policies[ref["name"]] = self._policy(ref, device)
            opp.choose = lambda meta: ref["name"]
        infos = first_rounds(self.vec, policy, device, deterministic=False, opponents=opp)
        points = sum(1.0 if i["success"] else 0.0 if i["deaths"] > 0 else 0.5 for i in infos)  # time ran out: a draw
        return points, len(infos)

    def games(self):
        out = []
        if self.games_file.exists():
            for line in self.games_file.read_text().splitlines():
                g = json.loads(line)
                out.append((g["player"], g["ref"], g["points"], g["rounds"]))
        return out

    def rate(self, model, steps):
        """The model's Elo (with its interval) and its score against each reference, all ratings fitted together."""
        device = model.device
        fresh = []  # references made from the model now: its score against them says nothing yet
        if not any(r["kind"] == "snapshot" for r in self.references):
            name = f"auto@{steps // 1_000_000}M"
            model.save(self.folder / name)
            self.references = [dict(name=name, kind="snapshot", file=name + ".zip", elo=ANCHOR_ELO)] + [dict(name=b, kind="bot") for b in BOT_REFERENCES]
            self._write()
            fresh.append(name)
        me = f"model@{steps}"
        scores = {}
        recheck = len({g[0] for g in self.games()}) % RECHECK_EVERY == 0  # ratings so far
        skipped = []
        with open(self.games_file, "a") as f:
            for ref in self.references:
                last = ref.get("last")
                if ref["kind"] == "bot" and not recheck and last is not None and not (1 - SURE < last < SURE):
                    skipped.append(ref["name"])
                    continue
                total = rounds = 0.0
                for fight in FIGHTS:
                    pts, n = self._play(model.policy, ref, fight, device)
                    total, rounds = total + pts, rounds + n
                    f.write(json.dumps(dict(steps=steps, player=me, ref=ref["name"], fight=fight, points=pts, rounds=n)) + "\n")
                scores[ref["name"]] = total / rounds
                ref["last"] = round(total / rounds, 3)
        anchor = next(r["name"] for r in self.references if r["kind"] == "snapshot") if self.references else me
        fitted = fit(self.games(), anchor)
        for r in self.references:  # for showing; the fit is what counts
            r["elo"] = round(fitted[r["name"]][0], 1) if r["name"] in fitted else r.get("elo")
        self._write()
        elo, ci = fitted[me]
        result = dict(elo=round(elo, 1), ci=round(ci, 1), scores={k: round(v, 3) for k, v in scores.items()}, steps=steps, fresh=fresh, skipped=skipped)
        if outgrown(scores, self.references, fresh):
            result["added"] = self.add_snapshot(model, steps, elo)
        return result

    def _write(self):
        """references.json, replaced whole: a stop mid-write must not leave half a file."""
        tmp = self.index.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.references, indent=1))
        os.replace(tmp, self.index)

    def add_snapshot(self, model, steps, elo):
        """A frozen copy of the model now joins the references; past MAX_SNAPSHOTS the oldest after the anchor goes
        (its games stay in games.jsonl and keep counting in the fit)."""
        name = f"auto@{steps // 1_000_000}M"
        if any(r["name"] == name for r in self.references):
            return None
        model.save(self.folder / name)
        self.references.append(dict(name=name, kind="snapshot", file=name + ".zip", elo=round(elo, 1)))
        snaps = [r for r in self.references if r["kind"] == "snapshot"]
        gone = []
        while len(snaps) > MAX_SNAPSHOTS:
            g = snaps.pop(1)
            self.references.remove(g)
            self.policies.pop(g["name"], None)
            gone.append(g)
        self._write()  # the index first: a stop between the two leaves a stray file, not a reference without one
        for g in gone:
            (self.folder / g["file"]).unlink(missing_ok=True)
        return name
