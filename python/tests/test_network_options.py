"""The network options (train_sim --critic-memory, --normalize-inputs, --entities) start from an older model exactly:
the same moves (and, but for a new critic memory, the same values) for the same inputs. Then they can learn.

    python test_network_options.py
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

torch.manual_seed(0)
vec = SimVecEnv(4, curriculum.Curriculum().levels, seed=2)
obs = torch.as_tensor(vec.reset())
obs = obs + torch.rand_like(obs) * 0.3  # varied inputs, not only a round's first tick


def make(**net):
    return ta.SplitPPO(MemoryPolicy, vec, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs=ta.policy_kwargs(**net), device="cpu", seed=0)


def answers(model):
    p = model.policy
    lstm = p.lstm_actor
    zeros = (torch.zeros(lstm.num_layers, len(obs), lstm.hidden_size), torch.zeros(lstm.num_layers, len(obs), lstm.hidden_size))
    starts = torch.ones(len(obs))
    with torch.no_grad():
        d, _ = p.get_distribution(obs, zeros, starts)
        v = p.predict_values(obs, zeros, starts)
    return torch.cat([x.logits for x in d.distribution], dim=1), v


base = make()
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "base"
    base.save(path)
    logits0, value0 = answers(base)
    # Opponents and hotbar slots through their own networks: added through a layer that starts at zero.
    m = make(entities=True)
    ta.load_weights(m, str(path) + ".zip")
    logits, value = answers(m)
    assert torch.allclose(logits, logits0, atol=1e-5) and torch.allclose(value, value0, atol=1e-5)
    assert any(p.requires_grad for n, p in m.policy.named_parameters() if "entity_out" in n)
    # Standardized inputs: folded into the first layer, the answers stay.
    m = make()
    ta.load_weights(m, str(path) + ".zip")
    body = m.policy.features_extractor.body_mean.shape[0]
    m.policy.features_extractor.fold_normalization(torch.rand(body) * 2 - 1, torch.rand(body) + 0.25)
    logits, value = answers(m)
    assert torch.allclose(logits, logits0, atol=1e-4) and torch.allclose(value, value0, atol=1e-4), (logits - logits0).abs().max()
    m.policy.features_extractor.fold_normalization(torch.rand(body), torch.rand(body) + 0.5)  # measured again: still the same
    logits, value = answers(m)
    assert torch.allclose(logits, logits0, atol=1e-4) and torch.allclose(value, value0, atol=1e-4)
    # A model saved before the standardization existed loads with none.
    state = base.policy.state_dict()
    old = {k: v for k, v in state.items() if not k.endswith(("body_mean", "body_std"))}
    fresh = make()
    fresh.policy.load_state_dict(old, strict=False)
    fresh.policy.features_extractor.load_state_dict({k.split("features_extractor.", 1)[1]: v for k, v in old.items() if k.startswith("features_extractor.")})
    assert bool((fresh.policy.features_extractor.body_std == 1).all())
    # The value's own memory: the moves stay; while it warms up only the value learns.
    m = make(critic_memory=True)
    ta.load_weights(m, str(path) + ".zip")
    m.policy.lstm_critic.load_state_dict(m.policy.lstm_actor.state_dict())
    logits, _ = answers(m)
    assert torch.allclose(logits, logits0, atol=1e-5)
    ta.freeze_all_but_value(m.policy, True)
    learning = {n for n, p in m.policy.named_parameters() if p.requires_grad}
    assert learning and all(n.startswith(ta.CRITIC_PARTS) for n in learning), learning
    ta.freeze_all_but_value(m.policy, False)
    assert all(p.requires_grad for p in m.policy.parameters())
    assert "lstm_critic.weight_ih_l0" not in ta.saved_keys(str(path) + ".zip")
print("network options ok")
