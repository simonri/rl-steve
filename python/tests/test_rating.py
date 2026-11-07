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
# The bonus weight is the share of the duel curriculum still ahead.
cur_ = curriculum.Curriculum()
assert ta.bonus_weight_for(cur_) == 1.0 - cur_.total("duel") / cur_.most("duel")
for d, values in curriculum.FAMILIES["duel"].items():
    cur_.levels["duel"][d] = len(values) - 1
assert ta.bonus_weight_for(cur_) == 0.0

# The joint fit: one that wins 10 to 1 against the anchor is about 400 above it; one that only ever met that one, and
# won 10 to 1 against it, about 400 above that, with a wider interval (it is known only through the other).
games = [("a", "anchor", 1000.0, 1100), ("b", "a", 1000.0, 1100)]
f = rating.fit(games, "anchor")
assert f["anchor"] == (1000.0, 0.0)
assert abs(f["a"][0] - 1400) < 15 and abs(f["b"][0] - 1800) < 30, f
assert f["b"][1] > f["a"][1] > 0, f
# A clean sweep stays finite: the weak prior keeps it to "much better".
assert rating.fit([("c", "anchor", 64.0, 64)], "anchor")["c"][0] < 3000

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
    # Every game is kept, one line per reference and kind of fight, and the rating comes with its interval.
    lines = (Path(d) / "references" / "games.jsonl").read_text().splitlines()
    assert len(lines) == len(refs) * len(rating.FIGHTS) and result["ci"] > 0, (len(lines), result)
    # A model that outgrows the references adds a frozen copy at its Elo; the anchor stays, the oldest others go.
    r = rating.Rating(Path(d), n=4)
    for m in range(10, 10 + rating.MAX_SNAPSHOTS + 1):
        r.add_snapshot(model, m * 1_000_000, 1000.0 + 100 * m)
    snaps = [x for x in r.references if x["kind"] == "snapshot"]
    assert len(snaps) == rating.MAX_SNAPSHOTS and snaps[0]["name"] == refs[0]["name"], [x["name"] for x in snaps]
    assert snaps[-1]["elo"] == 1000.0 + 100 * (10 + rating.MAX_SNAPSHOTS) and (Path(d) / "references" / snaps[-1]["file"]).exists()
    assert len(list((Path(d) / "references").glob("*.zip"))) == rating.MAX_SNAPSHOTS
    assert rating.Rating(Path(d), n=4).references == r.references  # kept across restarts
print("rating ok", result["elo"])
