"""Rating against frozen references: the Elo math, and a short run that adds the anchor and rates every reference.

    python test_rating.py
"""
import _path  # noqa: F401  (python/ on the import path)
import json
import tempfile
from pathlib import Path

import torch

import auto_env
import curriculum
import rating
import train_auto as ta
from sim_env import SimVecEnv
from spatial import MemoryPolicy

# Even against a 1000 is 1000; 10 to 1 odds is 400 points; a fit over several references lands in between.
assert abs(rating.elo_from_score(0.5, 1000) - 1000) < 1e-9
assert abs(rating.elo_from_score(10 / 11, 1000) - 1400) < 1e-6
assert abs(rating.fit_elo([(0.5, 1000, 32)]) - 1000) < 0.5
mid = rating.fit_elo([(10 / 11, 1000, 32), (1 / 11, 1800, 32)])
assert abs(mid - 1400) < 1, mid
# A clean sweep is "much better", not 4000: it counts as half a round short of one.
import math  # noqa: E402

assert rating.fit_elo([(1.0, 1000, 32)]) < 1000 + 400 * math.log10((1 - 0.5 / 32) / (0.5 / 32)) + 1  # about 1720
# A new frozen copy joins only when the model is well past the newest copy, whatever the bots say; not for one made in
# this very rating.
refs_ = [dict(name="a", kind="snapshot"), dict(name="b", kind="snapshot"), dict(name="walker", kind="bot")]
assert not rating.outgrown(dict(a=0.95, b=0.5, walker=1.0), refs_)
assert rating.outgrown(dict(a=0.95, b=0.8, walker=0.2), refs_)
assert rating.outgrown(dict(a=0.95, b=0.5), refs_, fresh=("b",)), "b made now: a is the newest copy that counts"
# The bonus weight reads only copies it did not just make, and follows the average of WEIGHT_OVER ratings.
assert ta.snapshot_score(dict(scores=dict(a=0.5, b=0.9), fresh=["b"]), refs_) == 0.5
assert ta.snapshot_score(dict(scores=dict(walker=1.0)), refs_) is None
assert ta.weight_for(0.5) == 1.0 and ta.weight_for(0.9) == 0.0 and abs(ta.weight_for(0.7) - 0.5) < 1e-9

# A short run: the anchor is added once (a frozen copy, Elo 1000), every bot gets a fixed rating, the model a score each.
rating.ROUNDS = 4
auto_env.DUEL_TICKS = 100  # short fights: what is checked here is the bookkeeping, not who wins
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # the model is slow on the processor a tick at a time
vec = SimVecEnv(1, curriculum.Curriculum().levels, seed=0)
model = ta.SplitPPO(MemoryPolicy, vec, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs=ta.POLICY, device=DEVICE, seed=0)
ta.fast_convolutions()
with tempfile.TemporaryDirectory() as d:
    r = rating.Rating(Path(d), n=4)
    result = r.rate(model, 5_000_000)
    refs = json.loads((Path(d) / "references" / "references.json").read_text())
    assert refs[0]["kind"] == "snapshot" and refs[0]["elo"] == 1000 and (Path(d) / "references" / refs[0]["file"]).exists()
    assert [x["name"] for x in refs[1:]] == list(rating.BOT_REFERENCES)
    assert set(result["scores"]) == {x["name"] for x in refs} and all(0 <= v <= 1 for v in result["scores"].values())
    assert rating.Rating(Path(d), n=4).references == refs  # a restart reads them back as they were, ratings and all
    # A model that outgrows the references adds a frozen copy at its Elo; the anchor stays, the oldest others go.
    r = rating.Rating(Path(d), n=4)
    for m in range(10, 10 + rating.MAX_SNAPSHOTS + 1):
        r.add_snapshot(model, m * 1_000_000, 1000.0 + 100 * m)
    snaps = [x for x in r.references if x["kind"] == "snapshot"]
    assert len(snaps) == rating.MAX_SNAPSHOTS and snaps[0]["name"] == refs[0]["name"], [x["name"] for x in snaps]
    assert snaps[-1]["elo"] == 1000.0 + 100 * (10 + rating.MAX_SNAPSHOTS) and (Path(d) / "references" / snaps[-1]["file"]).exists()
    assert len(list((Path(d) / "references").glob("*.zip"))) == rating.MAX_SNAPSHOTS
    assert rating.Rating(Path(d), n=4).references == r.references  # kept across restarts
    # The bonus weight reads only the copies a rating fought (one added just after has no score yet).
    assert ta.bonus_weight_from(dict(scores={refs[0]["name"]: 0.5}), r.references) == 1.0
print("rating ok", result["elo"])
