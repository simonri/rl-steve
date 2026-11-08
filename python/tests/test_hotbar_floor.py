"""The hotbar key's exploration floor (spatial.WeightedMultiCategorical.floor): every key keeps at least floor/10 of
probability while training, the other actions are untouched, learning runs, and a saved and loaded model has none.

    python test_hotbar_floor.py
"""
import _path  # noqa: F401  (python/ on the import path)
import tempfile
from pathlib import Path

import torch

import curriculum
import train_auto as ta
from sim_env import SimVecEnv
from spatial import MemoryPolicy

vec = SimVecEnv(4, curriculum.Curriculum().levels, seed=5)
m = ta.SplitPPO(MemoryPolicy, vec, n_steps=16, batch_size=32, n_epochs=1, policy_kwargs=ta.policy_kwargs(), device="cpu", seed=0)
p = m.policy
obs = torch.as_tensor(vec.reset())
lstm = p.lstm_actor
zeros = (torch.zeros(1, 4, lstm.hidden_size), torch.zeros(1, 4, lstm.hidden_size))
with torch.no_grad():
    p.action_net.bias[-10:] = torch.tensor([30.0] + [0.0] * 9)  # a model that never presses a key
    before, _ = p.get_distribution(obs, zeros, torch.ones(4))
    before = [d.probs.clone() for d in before.distribution]
    p.action_dist.floor = 0.05
    after, _ = p.get_distribution(obs, zeros, torch.ones(4))
assert float(before[-1][:, 1:].max()) < 1e-10
assert torch.allclose(after.distribution[-1].probs[:, 1:], torch.full((4, 9), 0.005), atol=1e-6), "each key at least floor / 10"
assert torch.allclose(after.distribution[-1].probs.sum(-1), torch.ones(4))
assert all(torch.equal(b, a.probs) for b, a in zip(before[:-1], after.distribution[:-1])), "the other actions untouched"
m.learn(16 * 4)
assert torch.isfinite(torch.tensor(m.logger.name_to_value["train/loss"]))
with tempfile.TemporaryDirectory() as tmp:
    m.save(Path(tmp) / "f")
    assert ta.load_policy(Path(tmp) / "f.zip", "cpu").action_dist.floor == 0.0, "the game plays without the floor"
print("hotbar floor ok")
