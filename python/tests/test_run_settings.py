"""Run settings from the OpenAI Five comparison (docs/training-design.md#fem): the discount set per run reaches the
aim potential, a new rule phases in, a reshaped model starts at learning rate 0, the rating skips bots it is sure of,
and a run can stop on the clock.

    python test_run_settings.py
"""
import _path  # noqa: F401  (python/ on the import path)
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import auto_env
import curriculum
import mcsim
import rating
import train_auto as ta
from sim_env import SimVecEnv
from train_sim import Deadline

# ---- The discount reaches the aim potential, and nothing else changes ----
levels = curriculum.Curriculum().levels
levels["duel"].update(bot=0)


def aim_points(gamma, ticks=60):
    vec = SimVecEnv(8, levels, seed=5)
    vec.env_method("set_exam", "duel", True)
    if gamma is not None:
        vec.batch.set_gamma(gamma)
    vec.reset()
    rng = np.random.default_rng(0)
    total = np.zeros(8)
    for _ in range(ticks):
        acts = np.stack([rng.integers(0, n, 8) for n in vec.action_space.nvec], 1)
        _, r, _, _ = vec.step(acts)
        total += r
    return total


same = aim_points(None), aim_points(mcsim.GAMMA)
assert np.array_equal(*same), "the default is shaping.rs GAMMA"
other = aim_points(0.9)
assert not np.allclose(same[0], other), "another discount changes the aim points"
for bad in (1.0, -0.1):
    try:
        mcsim.Batch(1).set_gamma(bad)
        raise AssertionError(f"gamma {bad} accepted")
    except ValueError:
        pass
assert auto_env.AutoEnv.gamma == mcsim.GAMMA

# ---- A new rule ramps in; a reshaped model learns nothing at first ----
with tempfile.TemporaryDirectory() as tmp:
    vec = SimVecEnv(2, levels, seed=1)
    p = ta.Progress(vec, Path(tmp), curriculum.Curriculum())
    p.model = SimpleNamespace(lr_schedule=None, ent_coef=None, n_steps=1)
    p.num_timesteps = 1_000_000
    p.phase_in("random_hand", 0.3, 1_000_000)
    assert vec.get_attr("random_hand") == [0.0, 0.0]
    p.num_timesteps = 1_500_000
    p._on_rollout_start()
    assert all(abs(v - 0.15) < 1e-9 for v in vec.get_attr("random_hand"))
    p.num_timesteps = 9_000_000
    p._on_rollout_start()
    assert all(abs(v - 0.3) < 1e-9 for v in vec.get_attr("random_hand")), "full, and no further"
    p.still_until = 9_100_000
    p._on_rollout_start()
    assert p.model.lr_schedule(1.0) == 0.0
    p.num_timesteps = 9_100_000
    p._on_rollout_start()
    assert p.model.lr_schedule(1.0) > 0.0

# ---- The rating skips bots it is sure of, but not every time, and never the frozen copies ----
with tempfile.TemporaryDirectory() as tmp:
    r = rating.Rating(tmp, n=4)
    played = []
    sure = {"slow walker": 1.0, "medium walker": 0.97, "walker": 0.8}

    def fake_play(policy, ref, fight, device):
        played.append(ref["name"])
        return sure.get(ref["name"], 0.5) * 4, 4

    r._play = fake_play
    model = SimpleNamespace(device="cpu", policy=None, save=lambda path: Path(str(path) + ".zip").write_bytes(b""))
    first = r.rate(model, 1_000_000)
    assert first["skipped"] == [], "nothing known yet"
    played.clear()
    second = r.rate(model, 2_000_000)
    assert set(second["skipped"]) == {"slow walker", "medium walker"}, second["skipped"]
    assert "walker" in played and any(n.startswith("auto@") for n in played)
    for steps in (3_000_000, 4_000_000, 5_000_000):  # the fifth rating, after four
        played.clear()
        last = r.rate(model, steps)
    assert last["skipped"] == [] and "slow walker" in played, "every RECHECK_EVERY-th rating plays them all"

# ---- The clock ----
d = Deadline(0.0)
time.sleep(0.01)
assert d._on_step() is False and Deadline(1.0)._on_step() is True
print("run settings ok")
