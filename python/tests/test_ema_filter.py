"""The weight average (SplitPPO.update_ema, save_ema) and advantage filtering (ShortStretches.keep_quantile), both from
Ataraxos (arXiv 2511.07312).

    python test_ema_filter.py
"""
import _path  # noqa: F401  (python/ on the import path)
import tempfile
from pathlib import Path

import numpy as np
import torch

import curriculum
import train_auto as ta
from sim_env import SimVecEnv
from spatial import MemoryPolicy

vec = SimVecEnv(4, curriculum.Curriculum().levels, seed=4)


def make():
    return ta.SplitPPO(MemoryPolicy, vec, n_steps=16, batch_size=32, n_epochs=1, policy_kwargs=ta.policy_kwargs(), device="cpu", seed=0)


# ---- The average: d * old + (1 - d) * new after each pass; saved as a model of its own; the model itself untouched ----
m = make()
m.ema_decay, m.ema = 0.5, {}
m.update_ema()
start = {k: v.clone() for k, v in m.policy.state_dict().items()}
m.learn(16 * 4)  # one pass, which updates the average
now = m.policy.state_dict()
name = next(k for k, v in now.items() if v.is_floating_point() and not torch.equal(v, start[k]))
assert torch.allclose(m.ema[name], 0.5 * start[name] + 0.5 * now[name], atol=1e-6), name
with tempfile.TemporaryDirectory() as tmp:
    m.save_ema(Path(tmp) / "ema")
    assert torch.equal(m.policy.state_dict()[name], now[name]), "saving the average leaves the model as it was"
    loaded = ta.load_policy(Path(tmp) / "ema.zip", "cpu").state_dict()
    assert torch.allclose(loaded[name], m.ema[name]), "the file holds the average"
    m.save(Path(tmp) / "plain")
    assert "ema" not in ta.saved_keys(str(Path(tmp) / "plain.zip")), "not inside the ordinary model file"

# ---- Filtering: only the top quarter of advantage sizes count, never an empty batch ----
m = make()
m.rollout_buffer.keep_quantile = 0.75
m.learn(16 * 4)
buf = m.rollout_buffer
assert abs(float(buf.keep.mean()) - 0.25) < 0.05, buf.keep.mean()
size = np.abs(buf.advantages)  # flattened in the same order as keep
assert (size[buf.keep > 0].min() >= size[buf.keep == 0].max()), "kept ones are the largest"
for s in buf.get(32):
    assert float(s.mask.sum()) > 0
assert np.isfinite(m.logger.name_to_value["train/value_loss"]) and abs(m.logger.name_to_value["train/kept"] - 0.25) < 0.05
print("ema and filter ok")
