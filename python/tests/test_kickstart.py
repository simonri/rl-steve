"""Kickstarting (train_auto.Kickstart): the teacher's choices are kept tick for tick in line with what the model saw,
the pull fades to nothing, and a model under the pull moves toward the teacher.

    python test_kickstart.py
"""
import _path  # noqa: F401  (python/ on the import path)
import copy

import torch

import curriculum
import train_auto as ta
from sim_env import SimVecEnv
from spatial import MemoryPolicy

torch.manual_seed(0)
vec = SimVecEnv(4, curriculum.Curriculum().levels, seed=3)


def make(seed, rate=3e-4):
    return ta.SplitPPO(MemoryPolicy, vec, n_steps=16, batch_size=32, n_epochs=1, learning_rate=rate, ent_coef=0.01,
                       policy_kwargs=ta.policy_kwargs(), device="cpu", seed=seed)


k = ta.Kickstart(make(0).policy, weight=2.0, over=100)
assert k.weight(0) == 2.0 and k.weight(50) == 1.0 and k.weight(100) == 0.0 and k.weight(500) == 0.0
assert k.width == sum(list(k.policy.action_space.nvec)[: ta.KICK_HEADS]), "every action but the hotbar key"

# A model that is its own teacher: each tick's stored teacher choices are exactly the model's own there, so the
# cross-entropy is the entropy of the taught actions. Off by even one tick or round, it would be larger.
m = make(0, rate=0.0)
m.rollout_buffer.kick = ta.Kickstart(copy.deepcopy(m.policy), 1.0, 10 ** 9)
m.rollout_buffer.reset()
m.learn(16 * 4)
policy, buf = m.policy, m.rollout_buffer
dists = []
to_dist = policy._get_action_dist_from_latent
policy._get_action_dist_from_latent = lambda latent: dists.append(to_dist(latent)) or dists[-1]
with torch.no_grad():
    for s in buf.get(32):
        policy.evaluate_actions(s.observations, s.actions, s.lstm_states, s.episode_starts)
        d = dists.pop()
        ce = ta.Kickstart.cross_entropy(buf.last_teacher, d)
        own = sum(x.entropy() for x in d.distribution[: ta.KICK_HEADS])
        mask = s.mask > 1e-8
        assert torch.allclose(ce[mask], own[mask], atol=1e-4), (ce[mask] - own[mask]).abs().max()
del policy._get_action_dist_from_latent

# Under a strong pull a fresh model moves toward another model's choices.
teacher = make(0).policy
with torch.no_grad():  # a teacher with firm, arbitrary choices (a new model leans to none, like the student)
    teacher.action_net.bias.copy_(torch.randn_like(teacher.action_net.bias) * 3)
student = make(1)
student.rollout_buffer.kick = ta.Kickstart(teacher, 5.0, 10 ** 9)
student.rollout_buffer.reset()
student.learn(16 * 4)
first = student.logger.name_to_value["train/kick_ce"]
student.learn(16 * 4 * 12, reset_num_timesteps=False)
last = student.logger.name_to_value["train/kick_ce"]
assert last < first - 0.1, (first, last)
print(f"kickstart ok {first:.2f} -> {last:.2f}")
