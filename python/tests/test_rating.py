"""Rating against frozen references: the Elo math, and a short run that adds the anchor and rates every reference.

    python test_rating.py
"""
import _path  # noqa: F401  (python/ on the import path)
import json
import tempfile
from pathlib import Path

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

# A short run: the anchor is added once (a frozen copy, Elo 1000), every bot gets a fixed rating, the model a score each.
rating.ROUNDS = 4
vec = SimVecEnv(1, curriculum.Curriculum().levels, seed=0)
model = ta.SplitPPO(MemoryPolicy, vec, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs=ta.POLICY, device="cpu", seed=0)
with tempfile.TemporaryDirectory() as d:
    r = rating.Rating(Path(d), n=4)
    result = r.rate(model, 5_000_000)
    refs = json.loads((Path(d) / "references" / "references.json").read_text())
    assert refs[0]["kind"] == "snapshot" and refs[0]["elo"] == 1000 and (Path(d) / "references" / refs[0]["file"]).exists()
    assert [x["name"] for x in refs[1:]] == list(rating.BOT_REFERENCES)
    assert set(result["scores"]) == {x["name"] for x in refs} and all(0 <= v <= 1 for v in result["scores"].values())
    again = rating.Rating(Path(d), n=4).rate(model, 6_000_000)  # the references stay as they were
    assert json.loads((Path(d) / "references" / "references.json").read_text()) == refs
print("rating ok", result["elo"])
