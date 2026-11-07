"""Moving a model into the grown network keeps it exactly as it was: with the movement model's inputs the same, its
moves and values come out the same whatever the new inputs say, and the new actions start where they should.

    python test_transfer.py [path to an older model, default auto6's latest]
"""
import _path  # noqa: F401  (python/ on the import path)
import sys
from pathlib import Path

import numpy as np
import torch

import curriculum
import mcsim
import train_auto as ta
from sim_env import SimVecEnv
from spatial import MemoryPolicy

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "models" / "auto6" / "latest.zip"
    torch.manual_seed(0)
    old = ta.SplitPPO.load(path, device="cpu")
    vec = SimVecEnv(4, curriculum.Curriculum().levels, seed=3)
    new = ta.SplitPPO(MemoryPolicy, vec, n_steps=8, batch_size=8, policy_kwargs=ta.POLICY, device="cpu")
    grew = ta.transfer(new.policy, old.policy.state_dict())
    print("grown:", grew)
    obs = vec.reset()
    old_obs = torch.as_tensor(obs[:, : mcsim.MOVE_OBS])
    new_obs = torch.as_tensor(obs.copy())
    new_obs[:, mcsim.MOVE_OBS :] = torch.rand(len(obs), mcsim.EXTRA) * 2 - 1  # anything at all in the new inputs
    starts = torch.ones(len(obs))
    with torch.no_grad():
        n_old = old.policy.lstm_actor.num_layers, len(obs), old.policy.lstm_actor.hidden_size
        zeros = lambda shape: (torch.zeros(shape), torch.zeros(shape))
        d_old, _ = old.policy.get_distribution(old_obs, zeros(n_old), starts)
        d_new, _ = new.policy.get_distribution(new_obs, zeros(n_old), starts)
        v_old = old.policy.predict_values(old_obs, zeros(n_old), starts)
        v_new = new.policy.predict_values(new_obs, zeros(n_old), starts)
    moves = len(d_old.distribution)
    for a, b in zip(d_old.distribution, d_new.distribution[:moves]):
        assert torch.allclose(a.logits, b.logits, atol=1e-5), (a.logits - b.logits).abs().max()
    assert torch.allclose(v_old, v_new, atol=1e-5), (v_old - v_new).abs().max()
    attack = d_new.distribution[moves].probs[:, 1]
    sword = 1 - d_new.distribution[moves + 1].probs[:, 0]  # any change of what is in hand
    assert torch.allclose(attack, attack[0]) and 0.05 < float(attack[0]) < 0.2, attack
    assert float(sword[0]) < 0.002, sword  # what the round started with stays in hand
    print(f"moves and values unchanged; new: attack {float(attack[0]):.2f}, change the slot {float(sword[0]):.4f}")
    print("transfer ok")


if __name__ == "__main__":
    main()
