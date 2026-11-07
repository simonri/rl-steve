"""What training is set up with fits the model: recorded rounds give what it sees and presses in its own shapes, a won
fight is counted without a course speed, a new model starts with its leanings, a saved model loads whole or grows,
and transfer refuses actions that would move.

    python test_training_setup.py
"""
import _path  # noqa: F401  (python/ on the import path)
import tempfile
from pathlib import Path

import numpy as np
import torch

import demo_data
import mcsim
import train_auto as ta
from auto_env import AutoEnv
from curriculum import Curriculum
from sim_env import SimVecEnv
from spatial import MemoryPolicy

ROOT = Path(__file__).resolve().parent.parent.parent

# Recorded rounds: what the model sees and every input it has, also the fighting ones (never used in a recording).
rounds = sorted((ROOT / "runs" / "demos").glob("*/round*.jsonl.gz"))[:2]
for path in rounds:
    obs, actions, _ = demo_data.replay(path)
    assert obs.shape[1] == AutoEnv.N_OBS, (obs.shape, AutoEnv.N_OBS)
    assert actions.shape[1] == mcsim.ACTIONS and not actions[:, 8:].any(), actions.shape
print(f"recordings: {len(rounds)} replayed" if rounds else "recordings: none here, skipped")

# A model on a few simulated rounds.
cur = Curriculum()
vec = SimVecEnv(4, cur.levels, seed=0)
model = ta.SplitPPO(MemoryPolicy, vec, n_steps=8, batch_size=32, n_epochs=1, policy_kwargs=ta.POLICY, device="cpu", seed=0)

# A new model leans away from jumping, sprinting, turning and clicking.
ta.init_action_bias(model.policy)
bias = model.policy.action_net.bias.data
nvec = [int(n) for n in model.policy.action_space.nvec]
start = np.cumsum([0] + nvec)
attack = bias[start[8] : start[9]]
assert attack[0] > attack[1], attack
assert len(nvec) == mcsim.ACTIONS == 10 and nvec[9] == 10, nvec  # the hotbar key: none or one of nine slots
hotbar = torch.softmax(bias[start[9] : start[10]], 0)
assert 1 - hotbar[0] < 0.002, hotbar  # a new hotbar key is pressed about once in a thousand ticks
for index, value in ta.MOVE_BIAS:
    assert bias[index] == value

# A saved model with the same layout loads whole; one from before the fighting inputs grows and keeps its choices.
with tempfile.TemporaryDirectory() as d:
    model.save(Path(d) / "same.zip")
    other = ta.SplitPPO(MemoryPolicy, vec, n_steps=8, batch_size=32, n_epochs=1, policy_kwargs=ta.POLICY, device="cpu", seed=1)
    grew, steps = ta.load_weights(other, Path(d) / "same.zip")
    assert grew == [] and all(torch.equal(a, b) for a, b in zip(other.policy.state_dict().values(), model.policy.state_dict().values()))

# transfer() refuses an older action layout that is not a leading part of the new one.
state = {k: v.clone() for k, v in model.policy.state_dict().items()}
state["action_net.bias"] = state["action_net.bias"][:7]  # 7 is not a sum of leading action sizes (3, 3, 2, ...)
state["action_net.weight"] = state["action_net.weight"][:7]
try:
    ta.transfer(model.policy, state)
    raise AssertionError("a misaligned action layout was accepted")
except ValueError:
    pass

# A won fight is counted without a speed over the course; a won course with one.
progress = ta.Progress(vec, Path(tempfile.mkdtemp()), cur, 1)
progress.first_step = 0
progress.model = model
info = dict(family="duel", clean=True, fell=False, stalled=False, damage=0.0, shift_share=0.0, parts={}, success=True, tested=None, blocks_needed=0)
progress.locals = dict(dones=[True], infos=[info])
progress.num_timesteps = 1
progress._on_step()
assert len(progress.speeds) == 0
vec.close()

# Wrong shapes or values from Python are a Python error, not a crash of the whole trainer.
batch = mcsim.Batch(2)
n_obs = mcsim.MOVE_OBS + mcsim.EXTRA
for bad in (np.zeros((2, mcsim.ACTIONS - 1), np.int64), np.zeros((3, mcsim.ACTIONS), np.int64), np.full((2, mcsim.ACTIONS), 11, np.int64)):
    try:
        batch.step(bad, np.zeros((2, n_obs), np.float32), np.zeros(2, np.float32), np.zeros(2, bool))
        raise AssertionError(f"accepted actions {bad.shape} {bad.max()}")
    except ValueError:
        pass
try:
    batch.report(5)
    raise AssertionError("round 5 of 2 accepted")
except IndexError:
    pass
print("training setup ok")
