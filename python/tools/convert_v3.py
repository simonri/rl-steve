"""Turns a model saved before v3 (4765 inputs, 10 actions with the hotbar) into v3 (4759 inputs, 9 actions): the
attack lock's input and the hotbar's history go out of the body's first layer, the hotbar's three choices out of the
action layer. Everything else is copied as it is, so with those inputs at 0 (the lock always was) it answers exactly
as before. The old file is kept next to it.

    python tools/convert_v3.py ../runs/models/auto7/latest.zip
"""
import _path  # noqa: F401  (python/ on the import path)
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import curriculum
import mcsim
import train_auto as ta
from auto_env import CUBE_START, HISTORY_START
from sim_env import SimVecEnv
from spatial import MemoryPolicy

path = Path(sys.argv[1])
old = ta.SplitPPO.load(path, device="cpu")
state, steps = old.policy.state_dict(), old.num_timesteps
assert old.policy.observation_space.shape[0] == 4765 and len(old.policy.action_space.nvec) == 10, "not a v2 model"
# In the body's input (the values before the cube, then everything from the history on): the extra part starts after
# 43 + 40 values. v2 extra: 60 enemy values, self 10 (index 2 the attack lock), command 4, then (attack, hotbar) x 5.
extra0 = CUBE_START + 40
dropped = [extra0 + 62] + [extra0 + 74 + 2 * t + 1 for t in range(5)]
keep_cols = [i for i in range(CUBE_START + 40 + 84) if i not in dropped]
vec = SimVecEnv(2, curriculum.Curriculum().levels, seed=0)
new = ta.SplitPPO(MemoryPolicy, vec, n_steps=8, batch_size=16, n_epochs=1, policy_kwargs=ta.POLICY, device="cpu", seed=0)
target = new.policy.state_dict()
for name, t in state.items():
    if name.endswith("body.0.weight"):
        t = t[:, keep_cols]
    elif name.startswith("action_net"):
        t = t[:38]
    assert t.shape == target[name].shape, (name, t.shape, target[name].shape)
    target[name] = t.clone()
new.policy.load_state_dict(target)
new.num_timesteps = steps

# Same answers: v2 inputs with the dropped ones at 0 against the v3 inputs without them.
rng = np.random.default_rng(0)
obs_new = rng.normal(0, 0.5, (8, mcsim.MOVE_OBS + mcsim.EXTRA)).astype(np.float32)
obs_old = np.zeros((8, 4765), np.float32)
body_new = list(range(CUBE_START)) + list(range(HISTORY_START, obs_new.shape[1]))
body_old = list(range(CUBE_START)) + list(range(HISTORY_START, 4765))
obs_old[:, :HISTORY_START] = obs_new[:, :HISTORY_START]
obs_old[:, [body_old[i] for i in keep_cols]] = obs_new[:, [body_new[i] for i in range(len(keep_cols))]]
with torch.no_grad():
    lstm = (torch.zeros(1, 8, 256), torch.zeros(1, 8, 256))
    starts = torch.ones(8)
    d_old, _ = old.policy.get_distribution(torch.tensor(obs_old), lstm, starts)
    d_new, _ = new.policy.get_distribution(torch.tensor(obs_new), lstm, starts)
for a, b in zip(d_old.distribution[:9], d_new.distribution):
    assert torch.allclose(a.logits, b.logits, atol=1e-5), (a.logits - b.logits).abs().max()
backup = path.with_name(path.stem + "_v2.zip")
shutil.copy(path, backup)
new.save(path)
print(f"{path}: v3 at {steps} decisions, same answers; the v2 file is {backup}")
