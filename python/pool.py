"""Self-play: who the model fights in training duels, and the pool of its older versions.

Why it is built this way, with sources: docs/training-design.md#self-play and #svagare. In short:

- Only the latest model as opponent can go round in circles: each version beats the last one and forgets how to
  beat older ones. So most self-play rounds are against the latest model, and the rest against a pool of frozen
  older versions (OpenAI Five: 80 % latest, 20 % past).
- Older versions are picked by how hard they are for the model now (prioritized fictitious self-play, AlphaStar):
  weight (1 − x)², x the model's score against it, so the ones it still loses to come up most. A tenth of the picks
  are uniform, so no version is forgotten entirely.
- Weaker opponents are older versions, not the latest one with random moves mixed in: an older version plays a
  real, coherent style; random moves only make a player that does not exist.
- The scripted bots stay: while the duel curriculum is unfinished, half of the duel rounds are against them (the
  curriculum learns only from those), then a tenth.

Only the model's own experience is learned from (opponents.py): the opponents just play.
"""
import json
import os
from pathlib import Path

import numpy as np

from opponents import BOT

POOL_EVERY = 5_000_000  # decisions between snapshots: about half an hour of training
MAX_POOL = 30  # versions kept; past that the oldest goes, except the first
LATEST_SHARE = 0.8  # of self-play rounds, against the latest model; the rest against the pool
UNIFORM_FLOOR = 0.1  # of pool picks, uniform rather than by difficulty
BOT_SHARE_LEARNING = 0.5  # of training duels, against scripted bots while the duel curriculum is unfinished
BOT_SHARE_DONE = 0.1  # and after
DECAY = 0.98  # each new result counts this much more than the one before: a version's score follows the model


def pfsp_weights(scores):
    """Pick weights for pool versions from the model's score against each (0 = always loses, 1 = always wins)."""
    x = np.asarray(scores, np.float64)
    hard = (1.0 - x) ** 2
    hard = hard / hard.sum() if hard.sum() > 0 else np.full(len(x), 1.0 / len(x))
    return (1.0 - UNIFORM_FLOOR) * hard + UNIFORM_FLOOR / len(x)


class Pool:
    """Frozen older versions of the model in runs/models/<run>/pool/, with the model's recent score against each."""

    def __init__(self, run_folder, seed=0):
        self.folder = Path(run_folder) / "pool"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index = self.folder / "pool.json"
        saved = json.loads(self.index.read_text()) if self.index.exists() else {}
        self.versions = saved.get("versions", [])  # [{name, file, steps, games, points}]
        self.last = saved.get("last")  # steps at the last snapshot
        self.rng = np.random.default_rng(seed)

    def save(self):
        tmp = self.folder / "pool.tmp"
        tmp.write_text(json.dumps(dict(versions=self.versions, last=self.last), indent=1))
        os.replace(tmp, self.index)

    def score(self, v):
        """The model's recent score against version v; 0.5 before any result (a prior of one draw)."""
        return (v["points"] + 0.5) / (v["games"] + 1.0)

    def due(self, steps):
        return self.last is None or steps - self.last >= POOL_EVERY

    def add(self, model, steps):
        """A frozen copy of the model now."""
        name = f"auto@{steps / 1e6:.0f}M"
        model.save(self.folder / name)
        self.versions = [v for v in self.versions if v["name"] != name]
        self.versions.append(dict(name=name, file=name + ".zip", steps=steps, games=0.0, points=0.0))
        while len(self.versions) > MAX_POOL:
            gone = self.versions.pop(1)
            (self.folder / gone["file"]).unlink(missing_ok=True)
        self.last = steps
        self.save()
        return name

    def pick(self):
        """A version to fight, by difficulty (PFSP); None if the pool is empty."""
        if not self.versions:
            return None
        w = pfsp_weights([self.score(v) for v in self.versions])
        return self.versions[int(self.rng.choice(len(self.versions), p=w))]["name"]

    def record(self, name, points):
        """A finished round against version `name`: the model's points (1 won, 0.5 draw, 0 lost)."""
        for v in self.versions:
            if v["name"] == name:
                v["games"] = DECAY * v["games"] + 1.0
                v["points"] = DECAY * v["points"] + points
                return True
        return False


class SelfPlay:
    """Opponents.choose for training: who plays a new duel round's opponent."""

    def __init__(self, opponents, pool, latest_policy, duel_finished, seed=0):
        self.opp, self.pool = opponents, pool
        self.opp.policies["latest"] = latest_policy  # the model being trained, as it is now
        self.duel_finished = duel_finished  # () -> bool: the duel curriculum is at its top
        self.rng = np.random.default_rng(seed)
        self.loaded = []  # pool versions with a policy in memory, oldest use first

    def __call__(self, meta):
        bots = BOT_SHARE_DONE if self.duel_finished() else BOT_SHARE_LEARNING
        if self.rng.random() < bots:
            return BOT
        if self.rng.random() < LATEST_SHARE:
            return "latest"
        name = self.pool.pick()
        if name is None:
            return "latest"
        self._load(name)
        return name

    def _load(self, name):
        if name in self.opp.policies:
            return
        import train_auto as ta  # here: train_auto imports the trainer's modules
        device = self.opp.policies["latest"].device
        v = next(v for v in self.pool.versions if v["name"] == name)
        self.opp.policies[name] = ta.load_policy(self.pool.folder / v["file"], device)
        self.loaded.append(name)
        # Versions dropped from the pool, or more in memory than the pool holds: let go of the least recently loaded
        # that no round is playing now.
        keep = {v["name"] for v in self.pool.versions}
        playing = set(self.opp.players)
        for old in list(self.loaded):
            if (old not in keep or len(self.loaded) > MAX_POOL) and old not in playing and old != name:
                self.loaded.remove(old)
                self.opp.policies.pop(old, None)
