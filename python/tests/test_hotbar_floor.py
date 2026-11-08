"""The hotbar key's exploration floor (spatial.WeightedMultiCategorical.floor, SplitPPO.hotbar_floor): only rounds that
start without blocks in hand get it, every key keeps at least floor/10 there, learning uses the same floor each tick
was played with, and a saved and loaded model has none.

    python test_hotbar_floor.py
"""
import _path  # noqa: F401  (python/ on the import path)
import tempfile
from pathlib import Path

import numpy as np
import torch

import curriculum
import items
import train_auto as ta
from sim_env import SimVecEnv
from spatial import MemoryPolicy

N = 8
vec = SimVecEnv(N, curriculum.Curriculum().levels, seed=5)
vec.set_attr("hand", "random")  # courses start with a random slot: most without blocks in hand
m = ta.SplitPPO(MemoryPolicy, vec, n_steps=16, batch_size=64, n_epochs=1, learning_rate=0.0, policy_kwargs=ta.policy_kwargs(), device="cpu", seed=0)
p = m.policy
obs = torch.as_tensor(vec.reset())
# Which rounds explore: exactly those without blocks in hand that have some.
for i in range(N):
    st = vec.batch.player(i)
    hot, slot = st["hotbar"], st["slot"]
    want = any(items.placeable(k) and c > 0 for k, c in hot) and not (items.placeable(hot[slot][0]) and hot[slot][1] > 0) and vec.meta[i]["family"] != "duel"
    assert bool(vec.explore[i]) == want or vec.meta[i]["family"] == "duel", (i, vec.meta[i]["family"])
assert vec.explore.any() and not vec.explore.all(), vec.explore

# A floor per row: the rows with one get at least floor / 10 per key, the others are untouched.
lstm = p.lstm_actor
zeros = (torch.zeros(1, N, lstm.hidden_size), torch.zeros(1, N, lstm.hidden_size))
with torch.no_grad():
    p.action_net.bias[-10:] = torch.tensor([30.0] + [0.0] * 9)  # a model that never presses a key
    p.action_dist.floor = torch.as_tensor(vec.explore.astype(np.float32) * 0.05).unsqueeze(1)
    d, _ = p.get_distribution(obs, zeros, torch.ones(N))
    p.action_dist.floor = 0.0
keys = d.distribution[-1].probs[:, 1:]
assert torch.allclose(keys[torch.as_tensor(vec.explore)], torch.full((int(vec.explore.sum()), 9), 0.005), atol=1e-6)
assert float(keys[~torch.as_tensor(vec.explore)].max()) < 1e-10

# Learning with the floors it played with: at learning rate 0 the new probabilities are exactly the old ones, so
# nothing is clipped. With the floor forgotten in learning, every explored tick would look like a huge change.
m.hotbar_floor = 0.05
m.rollout_buffer.floors = True
m.rollout_buffer.reset()
m.learn(16 * N)
used = m.rollout_buffer.floor_used
assert set(np.unique(used)) <= {0.0, np.float32(0.05)} and used.max() > 0, np.unique(used)
assert m.logger.name_to_value["train/clip_fraction"] == 0.0, m.logger.name_to_value["train/clip_fraction"]
assert abs(m.logger.name_to_value["train/approx_kl"]) < 1e-5
assert p.action_dist.floor == 0.0, "no floor left on the policy after a pass"
with tempfile.TemporaryDirectory() as tmp:
    m.save(Path(tmp) / "f")
    loaded = ta.load_policy(Path(tmp) / "f.zip", "cpu")
    assert loaded.action_dist.floor == 0.0, "the game plays without the floor"
    assert ta.SplitPPO.load(Path(tmp) / "f.zip", device="cpu", custom_objects=ta.QUIET_LOAD).hotbar_floor is None
print("hotbar floor ok")
