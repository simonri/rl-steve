"""How strong the model is at fighting, measured against frozen references it never trains against.

Why: a win rate against the opponents it trains with proves nothing, it can rise while the model goes round in
circles between styles (docs/training-design.md#referenser). So every so often the model fights a fixed set of
references in one fixed setting, and gets an Elo rating against their frozen ratings.

The references: the scripted bots, and frozen copies of the model saved in runs/models/<run>/references/. The first
frozen copy is the anchor, Elo 1000. A reference's own rating is set once, when it is added, from how the anchor does
against it, and never changes after. Ratings are Elo's usual scale: 400 points is ten to one odds.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import curriculum
import mcsim
from opponents import Opponents
from sim_env import SimVecEnv

BOT_REFERENCES = ("slow walker", "medium walker", "walker", "strafer", "combo", "pusher")
# One fixed fight for every reference, so ratings compare: a platform, 4 blocks apart, a little turned, wood swords.
FIGHT = dict(arena=0, distance=1, yaw=1, gear=0, kb=0, count=0)
ROUNDS = 32  # per reference and rating
ANCHOR_ELO = 1000.0
# When the model scores this much against the newest frozen copy of itself (the strongest reference), a new frozen copy
# joins the references, rated at the model's Elo then: against references it nearly always beats, a rating only says
# "much better" and swung 1350-1660 from rating to rating in auto7. 0.75 is about 190 Elo above it. Only the copies
# count: the bots, beaten ~95 %, carried an average over all references past the line against an equal copy, and a
# near-copy joined at every rating (auto7 210M, 212M, 214M).
ADD_AT = 0.75
MAX_SNAPSHOTS = 4  # the anchor and the three newest: each costs ROUNDS fights a rating


def elo_from_score(score, against):
    """The rating that scores `score` (wins + half the draws, per round) against a player rated `against`."""
    s = min(max(score, 0.02), 0.98)  # a clean sweep says only "much better", not how much
    return against + 400.0 * math.log10(s / (1.0 - s))


def _clamped(score, n):
    """A score as `n` rounds can tell it: a clean sweep counts as half a round short of one (else the fit runs off)."""
    return min(max(score, 0.5 / n), 1.0 - 0.5 / n)


def fit_elo(results):
    """The one rating that best explains scores against several fixed ratings: [(score, rating, rounds)]."""
    def loglik(r):
        total = 0.0
        for score, rating, n in results:
            score = _clamped(score, n)
            e = 1.0 / (1.0 + 10.0 ** ((rating - r) / 400.0))
            e = min(max(e, 1e-9), 1 - 1e-9)
            total += n * (score * math.log(e) + (1.0 - score) * math.log(1.0 - e))
        return total
    grid = np.arange(-1000.0, 4000.0, 5.0)
    best = grid[int(np.argmax([loglik(r) for r in grid]))]
    fine = np.arange(best - 5.0, best + 5.0, 0.25)
    return float(fine[int(np.argmax([loglik(r) for r in fine]))])


def outgrown(scores, references, fresh=()):
    """The model is well past the newest frozen copy of itself (ADD_AT), so a new copy should join the references. The
    bots do not count, nor a copy made in this very rating."""
    older = [r for r in references if r["kind"] == "snapshot" and r["name"] not in fresh]
    return bool(older) and scores.get(older[-1]["name"], 0.0) >= ADD_AT


class Rating:
    """For train_sim: rates the model against the references, adding the anchor the first time."""

    def __init__(self, run_folder, n=32):
        self.folder = Path(run_folder) / "references"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index = self.folder / "references.json"
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

    def _score(self, policy, ref, device):
        """Wins plus half the draws, per round, of `policy` fighting `ref` in the fixed fight."""
        levels = curriculum.Curriculum().levels
        levels["duel"].update(FIGHT, bot=curriculum.BOTS.index(ref["name"]) if ref["kind"] == "bot" else 0)
        self.vec.env_method("set_levels", levels)
        self.vec.env_method("set_exam", "duel", True)
        if ref["kind"] == "bot":
            self.vec.opponents = None
        else:
            opp = Opponents(self.vec.num_envs)
            opp.policies[ref["name"]] = self._policy(ref, device)
            opp.choose = lambda meta: ref["name"]
            self.vec.opponents = opp
        obs = self.vec.reset()
        lstm = policy.lstm_actor
        n = self.vec.num_envs
        memory = (torch.zeros(lstm.num_layers, n, lstm.hidden_size, device=device), torch.zeros(lstm.num_layers, n, lstm.hidden_size, device=device))
        starts = torch.ones(n, device=device)
        # Each of the n rounds counts once, its first: counting the first n to finish counted quick wins and losses
        # twice and left out the draws, which take the whole round.
        result = [None] * n
        with torch.no_grad():
            while any(r is None for r in result):
                dist, memory = policy.get_distribution(torch.as_tensor(obs, device=device), memory, starts)
                self.vec.step_async(dist.get_actions(deterministic=False).cpu().numpy())
                obs, _, dones, infos = self.vec.step_wait()
                starts = torch.as_tensor(dones, dtype=torch.float32, device=device)
                for i in np.flatnonzero(dones):
                    if result[i] is None:
                        info = infos[i]
                        result[i] = 1.0 if info["success"] else 0.0 if info["deaths"] > 0 else 0.5  # time ran out: a draw
        self.vec.opponents = None
        return sum(result) / n

    def rate(self, model, steps):
        """The model's Elo and its score against each reference; adds the anchor (a frozen copy of the model) first."""
        device = model.device
        fresh = []  # references made from the model now: its score against them says nothing yet
        if not any(r["kind"] == "snapshot" for r in self.references):
            name = f"auto@{steps // 1_000_000}M"
            model.save(self.folder / name)
            anchor = dict(name=name, kind="snapshot", file=name + ".zip", elo=ANCHOR_ELO)
            self.references = [anchor]
            for bot in BOT_REFERENCES:
                score = self._score(self._policy(anchor, device), dict(name=bot, kind="bot"), device)
                # The bot's rating from the anchor's score against it: elo_from_score gives the anchor's rating
                # relative to the bot, so the bot is that far the other way.
                self.references.append(dict(name=bot, kind="bot", elo=round(2 * ANCHOR_ELO - elo_from_score(score, ANCHOR_ELO), 1)))
            self._write()
            fresh.append(name)
        n = self.vec.num_envs
        scores = {r["name"]: self._score(model.policy, r, device) for r in self.references}
        elo = fit_elo([(scores[r["name"]], r["elo"], n) for r in self.references if r["name"] not in fresh])
        result = dict(elo=round(elo, 1), scores={k: round(v, 3) for k, v in scores.items()}, steps=steps, fresh=fresh)
        if outgrown(scores, self.references, fresh):
            result["added"] = self.add_snapshot(model, steps, elo)
        return result

    def _write(self):
        """references.json, replaced whole: a stop mid-write must not leave half a file."""
        tmp = self.index.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.references, indent=1))
        os.replace(tmp, self.index)

    def add_snapshot(self, model, steps, elo):
        """A frozen copy of the model now joins the references at Elo `elo`; past MAX_SNAPSHOTS the oldest after the
        anchor goes. Ratings already given stay as they were."""
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
