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


def elo_from_score(score, against):
    """The rating that scores `score` (wins + half the draws, per round) against a player rated `against`."""
    s = min(max(score, 0.02), 0.98)  # a clean sweep says only "much better", not how much
    return against + 400.0 * math.log10(s / (1.0 - s))


def fit_elo(results):
    """The one rating that best explains scores against several fixed ratings: [(score, rating, rounds)]."""
    def loglik(r):
        total = 0.0
        for score, rating, n in results:
            e = 1.0 / (1.0 + 10.0 ** ((rating - r) / 400.0))
            e = min(max(e, 1e-9), 1 - 1e-9)
            total += n * (score * math.log(e) + (1.0 - score) * math.log(1.0 - e))
        return total
    grid = np.arange(-1000.0, 4000.0, 5.0)
    best = grid[int(np.argmax([loglik(r) for r in grid]))]
    fine = np.arange(best - 5.0, best + 5.0, 0.25)
    return float(fine[int(np.argmax([loglik(r) for r in fine]))])


class Rating:
    """For train_sim: rates the model against the references, adding the anchor the first time."""

    def __init__(self, run_folder, n=32):
        self.folder = Path(run_folder) / "references"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index = self.folder / "references.json"
        self.references = json.loads(self.index.read_text()) if self.index.exists() else []
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
        points = rounds = 0
        with torch.no_grad():
            while rounds < ROUNDS:
                dist, memory = policy.get_distribution(torch.as_tensor(obs, device=device), memory, starts)
                self.vec.step_async(dist.get_actions(deterministic=False).cpu().numpy())
                obs, _, dones, infos = self.vec.step_wait()
                starts = torch.as_tensor(dones, dtype=torch.float32, device=device)
                for i in np.flatnonzero(dones):
                    rounds += 1
                    info = infos[i]
                    points += 1.0 if info["success"] else 0.0 if info["deaths"] > 0 else 0.5  # time ran out: a draw
        self.vec.opponents = None
        return points / rounds

    def rate(self, model, steps):
        """The model's Elo and its score against each reference; adds the anchor (a frozen copy of the model) first."""
        device = model.device
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
            self.index.write_text(json.dumps(self.references, indent=1))
        scores = {r["name"]: self._score(model.policy, r, device) for r in self.references}
        elo = fit_elo([(scores[r["name"]], r["elo"], ROUNDS) for r in self.references])
        return dict(elo=round(elo, 1), scores={k: round(v, 3) for k, v in scores.items()}, steps=steps)
