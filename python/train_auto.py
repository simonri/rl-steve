"""Trains on every course at once. curriculum.py moves the difficulty, so there are no stages to get stuck on."""
import argparse
import copy
import csv
import json
import os
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples, RNNStates
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.distributions import Distribution

import demo_data
import mcsim
import scoreboard as sb
from auto_env import NO_TURN, TURN_STEPS, AutoEnv
from spatial import MemoryPolicy, SpatialNet
from curriculum import FAMILIES, Curriculum, finished, is_open

ROOT = Path(__file__).resolve().parent.parent
EXAM_EVERY = 200_000
SAVE_EVERY = 20_000  # decisions between saves of latest.zip, which the live viewer picks up
EXAM_ROUNDS = 32  # per open course kind, best action only and every dial at its limit
WINDOW = 100
# One decision per tick, so these keep looking the same distance ahead in game time as before.
GAMMA, GAE_LAMBDA = 0.995, 0.975
# Learning calms down over this stretch, so what has been learned is not constantly overwritten.
CALM_FROM, CALM_OVER = 500_000, 5_000_000  # decisions into a run: the old 16M was meant for one run and left new ones wild
STILL_AFTER_SURGERY = 300_000  # decisions at learning rate 0 after a model was reshaped (Progress.still_until)
LEARNING_RATE, LEARNING_RATE_END = 3e-4, 3e-5
ENTROPY, ENTROPY_END = 0.01, 0.004  # the end was 0.002: late in a run new skills (fighting) were hardly explored
# Learning stops going over a stretch again once the way it plays has moved this far. With the memory, 8 passes moved
# it about 0.28 each time (clipping 56 % of everything), so far that what it had learned fell apart.
TARGET_KL = 0.02
# Eyes, then a memory carried between ticks, then one layer each for choosing and for judging how it is going.
# Judging has no memory of its own, which would cost as much again every tick. It reads the eyes directly instead of
# the memory, so what it learns about how a round is going also trains the eyes: with it reading the memory without
# being able to train anything before it, auto5 never learned to bridge in 250k decisions, where auto4 did in 50k.
POLICY = dict(features_extractor_class=SpatialNet, lstm_hidden_size=256, shared_lstm=False, enable_critic_lstm=False,
              net_arch=dict(pi=[256], vf=[256]), activation_fn=torch.nn.SiLU)
NAMES = {"hop": "Parkour", "path": "Path", "bridge": "Bridge", "treenan": "Treenan", "stairs": "Stairs", "bedwars": "Bedwars", "duel": "Duel"}


# A new left click starts at about one in nine. A new hotbar key (none, slot 1-9) starts at about one press in a
# thousand ticks, so a model that had no hotbar plays as before: a random change puts something else in hand
# (spatial.ENTROPY_WEIGHTS keeps exploration from pushing it back up).
NEW_ACTION_BIAS = {2: (1.0, -1.0), 10: (9.0,) + (0.0,) * 9}


def transfer(policy, old_state):
    """An older model's weights in a model whose inputs or actions grew. Same shapes are copied. A layer that takes more
    inputs keeps the old columns and starts the new ones at 0, so the old inputs give exactly the old result. More
    actions keep the old rows; the new ones start at 0 with a bias toward not attacking and keeping the hotbar slot.
    Returns the names of what grew."""
    new = policy.state_dict()
    grew = []
    for name, old in old_state.items():
        if name not in new:
            continue
        t = new[name]
        if t.shape == old.shape:
            new[name] = old.clone()
            continue
        if name.startswith("value_net") and getattr(policy, "hl_gauss", False):
            continue  # a value head of bins replaces a single number: it starts afresh (and warms up alone)
        if name.startswith("action_net") and t.shape[0] > old.shape[0] and sum(sizes_of(old.shape[0], policy)) != old.shape[0]:
            # Only actions added at the end keep their meaning; anything else would quietly move old choices around.
            raise ValueError(f"{name}: the older {old.shape[0]} choices are not the first actions of {list(policy.action_space.nvec)}")
        if name.endswith("action_net.weight") and t.shape[1] == old.shape[1] and t.shape[0] > old.shape[0]:
            w = torch.zeros_like(t)
            w[: old.shape[0]] = old
        elif name.endswith("action_net.bias") and t.shape[0] > old.shape[0]:
            w = torch.zeros_like(t)
            w[: old.shape[0]] = old
            sizes = list(policy.action_space.nvec)[len(sizes_of(old.shape[0], policy)):]
            i = old.shape[0]
            for n in sizes:
                w[i : i + n] = torch.tensor(NEW_ACTION_BIAS.get(int(n), (0.0,) * int(n)))
                i += int(n)
        elif t.dim() == 1 and t.shape[0] > old.shape[0] and name.endswith(("body_mean", "body_std")):
            w = t.clone()  # the inputs grew at the end: the old ones keep their standardization, the new start at none
            w[: old.shape[0]] = old
        elif t.dim() == 2 and t.shape[0] == old.shape[0] and t.shape[1] > old.shape[1]:
            w = torch.zeros_like(t)
            w[:, : old.shape[1]] = old
        else:
            raise ValueError(f"{name}: {tuple(old.shape)} does not fit {tuple(t.shape)}")
        new[name] = w
        grew.append(name)
    policy.load_state_dict(new)
    return grew


# A new model starts with rare jumps and sprints and a mostly steady view, so random inputs do not throw it off the
# platform before it has learned anything (indices into the action logits); the fighting inputs as NEW_ACTION_BIAS.
MOVE_BIAS = ((10, 1.5), (11, -1.5), (12, 1.5), (13, -1.5), (14 + NO_TURN, 1.5), (14 + len(TURN_STEPS) + NO_TURN, 1.5))
MOVE_ACTIONS = 8  # the movement model's inputs; the fighting ones come after them


def policy_kwargs(critic_memory=False, entities=False, normalize_inputs=False, hl_gauss=False):
    """POLICY with the network options a run was made with (train_sim keeps them in the run's net.json):
    critic_memory gives the value its own memory (an LSTM, as the policy has); entities gives opponents and hotbar slots
    their own small shared networks (spatial.SpatialNet); normalize_inputs needs no change here (SpatialNet always has
    its standardization, the identity until measured)."""
    kw = dict(POLICY)
    kw["enable_critic_lstm"] = bool(critic_memory)
    kw["hl_gauss"] = bool(hl_gauss)  # the value as bins (spatial.HLGaussValue)
    if entities:
        kw["features_extractor_kwargs"] = dict(kw.get("features_extractor_kwargs", {}), entities=True)
    return kw


def saved_keys(path):
    """The names of the weights a saved model has."""
    return set(SplitPPO.load(path, device="cpu", custom_objects=QUIET_LOAD).policy.state_dict())


# The value's own parts: while VALUE_WARMUP runs (a new critic memory), only these learn.
CRITIC_PARTS = ("lstm_critic", "mlp_extractor.value_net", "value_net")


def freeze_all_but_value(policy, on):
    """While `on`, only the value learns: a new critic memory starts out wrong, and its advantages would push the
    policy around until it has caught up. The policy, eyes and memory keep their weights."""
    for name, p in policy.named_parameters():
        p.requires_grad_(not on or name.startswith(CRITIC_PARTS))


@torch.no_grad()
def measure_inputs(model, vec, ticks=200, floor=0.25):
    """The mean and spread of what the body part sees, from `ticks` ticks of every round played by the model.
    Spreads below `floor` are raised to it: a value that is nearly always 0 (a rare item) must not be blown up."""
    from auto_env import CUBE_START, HISTORY_START

    policy = model.policy
    obs = vec.reset()
    n = vec.num_envs
    lstm = policy.lstm_actor
    memory = (torch.zeros(lstm.num_layers, n, lstm.hidden_size, device=model.device), torch.zeros(lstm.num_layers, n, lstm.hidden_size, device=model.device))
    starts = torch.ones(n, device=model.device)
    seen = []
    for _ in range(ticks):
        seen.append(np.concatenate([obs[:, :CUBE_START], obs[:, HISTORY_START:]], axis=1))
        dist, memory = policy.get_distribution(torch.as_tensor(obs, device=model.device), memory, starts)
        vec.step_async(dist.get_actions(deterministic=False).cpu().numpy())
        obs, _, dones, _ = vec.step_wait()
        starts = torch.as_tensor(dones, dtype=torch.float32, device=model.device)
    x = np.concatenate(seen)
    return torch.as_tensor(x.mean(0), dtype=torch.float32), torch.as_tensor(np.maximum(x.std(0), floor), dtype=torch.float32)


def init_action_bias(policy):
    """The starting leanings of a new model's choices (MOVE_BIAS, NEW_ACTION_BIAS)."""
    bias = policy.action_net.bias.data
    for index, value in MOVE_BIAS:
        bias[index] = value
    nvec = [int(n) for n in policy.action_space.nvec]
    i = sum(nvec[:MOVE_ACTIONS])
    for n in nvec[MOVE_ACTIONS:]:
        bias[i : i + n] = torch.tensor(NEW_ACTION_BIAS.get(n, (0.0,) * n))
        i += n


def load_weights(model, path):
    """A saved model's weights into `model`. The same layout loads whole, learning state included; an older one whose
    inputs or actions have since grown goes through transfer(), which keeps what it knew exactly. Loaded on the
    processor: a second model on the card takes the scratch memory the fast convolutions need. Returns (what grew, the
    saved model's step count)."""
    old = SplitPPO.load(path, device="cpu", custom_objects=QUIET_LOAD)
    state, steps = old.policy.state_dict(), old.num_timesteps
    del old
    new = model.policy.state_dict()
    if state.keys() == new.keys() and all(state[k].shape == new[k].shape for k in state):
        model.set_parameters(str(path), device=model.device)
        return [], steps
    return transfer(model.policy, state), steps


def sizes_of(total, policy):
    """The leading action sizes whose choices add up to `total` (the older model's actions)."""
    out, s = [], 0
    for n in policy.action_space.nvec:
        if s >= total:
            break
        out.append(int(n))
        s += int(n)
    return out


def fast_convolutions():
    """Stable-Baselines3 turns on cuDNN's deterministic mode when given a seed. That leaves only slow ways through the
    convolutions' backward pass: a learning step took 500 ms instead of 120. Call after the model is made."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_policy(path, device):
    """A saved model's policy only, to play with (opponents, references), on `device`.

    Loaded on the processor and then moved: SB3 sets up a loaded model with its seed, and on the graphics card that
    switches cuDNN to deterministic for the whole process, training included. That is what made learning 3.5x slower
    (75 -> 270 ms a step) from the moment the first pool version was loaded. fast_convolutions again, to be sure."""
    policy = SplitPPO.load(path, device="cpu", custom_objects=QUIET_LOAD).policy
    # One saved with an older layout (fewer inputs or actions, before the hotbar) plays through transfer(): the new
    # inputs at 0, the new actions at their starting leanings, everything it knew exactly as it was. Fed today's
    # inputs as it was, it crashed training at the first pool pick or rating (2x530 against 161x128).
    space, actions = policy.observation_space.shape[0], [int(n) for n in policy.action_space.nvec]
    if space != mcsim.MOVE_OBS + mcsim.EXTRA or actions != list(mcsim.CHOICES):
        import gymnasium as gym

        new = MemoryPolicy(gym.spaces.Box(-5.0, 5.0, (mcsim.MOVE_OBS + mcsim.EXTRA,), np.float32), gym.spaces.MultiDiscrete(list(mcsim.CHOICES)),
                           lambda _: 0.0, **POLICY)
        transfer(new, policy.state_dict())
        policy = new
    policy = policy.to(device)
    policy.set_training_mode(False)
    fast_convolutions()
    return policy


# For a load that only wants the weights: no reseeding of every random generator (SB3 seeds them all again from the
# saved seed, restarting training's action sampling at the same point with every pool or reference load), and the
# smallest rollout buffer it can build (a full one was most of a load's 730 MB and seconds).
QUIET_LOAD = {"seed": None, "n_steps": 1}


def make_env(port, levels):
    def build():
        env = AutoEnv(port, levels=levels)
        env.camera_every = 2.0  # rounds run far faster than anyone can follow here
        return Monitor(env, info_keywords=("success",))

    return build


# Learning follows the memory back at most this many ticks. It still starts each stretch from the memory the model
# really had there while playing, so it remembers further than this, it just learns from shorter stretches.
STRETCH = 32


KICK_HEADS = 9  # the hotbar key (the last action) is left out: the teacher never learnt to find its blocks


class Kickstart:
    """A teacher's moves as an extra pull on a model learning from scratch, fading to nothing (Schmitt et al. 2018,
    "Kickstarting Deep Reinforcement Learning", arXiv 1803.03835).

    Why: from scratch auto8 never found bridging (0 of 60 rounds at the first level after 10M decisions): a bridge pays
    only after crouching to the edge, looking back down and placing, in that order. A teacher shows how, on the
    rounds the new model itself is playing, which old recordings could not. The pull is the cross-entropy from the
    teacher's choices to the model's, weighted by `weight` at the start and falling linearly to 0 at `over` decisions:
    after that only the reward counts, so it keeps none of the teacher's habits it cannot justify. The hotbar key is
    not taught (KICK_HEADS): the teacher never takes its blocks out, and that is a habit to lose."""

    def __init__(self, teacher, weight=1.0, over=20_000_000):
        self.policy = teacher
        self.policy.set_training_mode(False)
        self.weight0, self.over = float(weight), int(over)
        self.width = int(sum(teacher.action_space.nvec[:KICK_HEADS]))
        self.state = None

    def weight(self, steps):
        return self.weight0 * max(0.0, 1.0 - steps / self.over)

    @torch.no_grad()
    def label(self, obs, starts):
        """The teacher's log-probabilities for these observations, its memory carried tick to tick like the model's."""
        p = self.policy
        lstm = p.lstm_actor
        if self.state is None or self.state[0].shape[1] != len(obs):
            zeros = torch.zeros(lstm.num_layers, len(obs), lstm.hidden_size, device=p.device)
            self.state = (zeros, zeros.clone())
        dist, self.state = p.get_distribution(torch.as_tensor(obs, device=p.device), self.state,
                                              torch.as_tensor(starts, dtype=torch.float32, device=p.device))
        return torch.cat([torch.log_softmax(d.logits, -1) for d in dist.distribution[:KICK_HEADS]], 1).cpu().numpy()

    @staticmethod
    def cross_entropy(teacher, dist):
        """Per tick: -sum over the taught actions of p_teacher * log p_model."""
        mine = torch.cat([torch.log_softmax(d.logits, -1) for d in dist.distribution[:KICK_HEADS]], 1)
        return -(teacher.exp() * mine).sum(-1)  # a padding row (teacher -inf) gives 0


class ShortStretches(RecurrentRolloutBuffer):
    """Hands the learner its batches as pieces of rounds, each padded to the longest piece.

    The library cuts a batch only where a round starts, so one long round next to a few short ones made the network go
    over many times more ticks than there were; this also cuts every STRETCH ticks. And it sent every piece to the
    graphics card on its own, thousands of small copies that took most of the learning time; this sends the whole
    stretch of play once and does the padding there."""

    kick = None  # a Kickstart: the teacher's choices are kept for every tick (train_sim --kick)

    def reset(self):
        super().reset()
        self.on_card = {}
        if self.kick is not None:
            self.teacher = np.zeros((self.buffer_size, self.n_envs, self.kick.width), np.float32)

    def add(self, *args, lstm_states, **kwargs):
        if self.kick is not None:  # args: observations, actions, rewards, episode starts, values, log-probs
            self.teacher[self.pos] = self.kick.label(args[0], args[3])
        super().add(*args, lstm_states=lstm_states, **kwargs)

    def get(self, batch_size=None):
        if self.kick is not None and not self.generator_ready:
            self.teacher = self.swap_and_flatten(self.teacher)  # in the same order as everything else
        yield from super().get(batch_size)

    def _card(self, name):
        if name not in self.on_card:
            self.on_card[name] = torch.as_tensor(self.__dict__[name], device=self.device)
        return self.on_card[name]

    def _get_samples(self, batch_inds, env_change, env=None):
        starts = np.logical_or(self.episode_starts[batch_inds].ravel() > 0, env_change[batch_inds].ravel() > 0)
        starts[::STRETCH] = True
        first = np.flatnonzero(starts)
        piece = np.cumsum(starts) - 1
        where = np.arange(len(batch_inds)) - first[piece]
        pieces, longest = len(first), int(where.max()) + 1
        picked = torch.as_tensor(batch_inds, device=self.device)
        piece, where = torch.as_tensor(piece, device=self.device), torch.as_tensor(where, device=self.device)

        def pad(data, fill=0.0):
            out = data.new_full((pieces, longest, *data.shape[1:]), fill)
            out[piece, where] = data
            return out.reshape(pieces * longest, *data.shape[1:])

        def column(name):
            return pad(self._card(name)[picked]).flatten()

        firsts = picked[torch.as_tensor(first, device=self.device)]
        states = [self._card(name)[firsts].swapaxes(0, 1).contiguous() for name in ("hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf")]
        self.last_returns = column("returns")  # for an HL-Gauss value's loss (SplitPPO.train)
        if self.kick is not None:
            # Padding rows get -inf: no teacher choice there, so they add nothing to the pull (SplitPPO.train).
            self.last_teacher = pad(self._card("teacher")[picked], float("-inf"))
        return RecurrentRolloutBufferSamples(
            # What it saw is most of a stretch's memory (600 MB for 512 rounds), so it goes to the card a batch at a
            # time. With all of it there the card ran short, and cuDNN fell back to a convolution 4x slower.
            observations=pad(torch.from_numpy(self.observations[batch_inds]).to(self.device), float("nan")),  # MemoryPolicy skips these
            actions=pad(self._card("actions")[picked]),
            old_values=column("values"),
            old_log_prob=column("log_probs"),
            advantages=column("advantages"),
            returns=column("returns"),
            lstm_states=RNNStates((states[0], states[1]), (states[2], states[3])),
            episode_starts=column("episode_starts"),
            mask=pad(torch.ones(len(batch_inds), device=self.device)),
        )


class SplitPPO(RecurrentPPO):
    """Learns on the graphics card and plays on the processor. Learning works on big batches, which a graphics card is
    fast at. Playing picks one action at a time, where moving the data to the card costs more than it saves.
    The memory it carries between ticks stays on the processor too, since that is where it plays."""

    def _excluded_save_params(self):
        # What the games looked like when it stopped is no use later, and with hundreds of simulated rounds it is most
        # of the file. Learning starts the games afresh when these are missing.
        return super()._excluded_save_params() + ["actor", "_last_obs", "_last_original_obs", "_last_episode_starts", "_last_lstm_states", "ep_info_buffer", "ep_success_buffer"]

    def _setup_model(self):
        super()._setup_model()
        old = self.rollout_buffer
        self.rollout_buffer = ShortStretches(self.n_steps, self.observation_space, self.action_space, old.hidden_state_shape, self.device,
                                             gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.n_envs)

    @contextmanager
    def playing(self):
        # Hundreds of simulated rounds at once are a big batch too, so then playing stays on the card as well.
        if self.device.type == "cpu" or getattr(self, "play_on_card", False):
            yield
            return
        learner, device = self.policy, self.device
        if getattr(self, "actor", None) is None:
            self.actor = copy.deepcopy(learner).to("cpu")
        self.actor.load_state_dict({k: v.to("cpu") for k, v in learner.state_dict().items()})
        self.actor.set_training_mode(False)
        self.policy, self.device = self.actor, torch.device("cpu")
        self._last_lstm_states = RNNStates(*(tuple(t.cpu() for t in part) for part in self._last_lstm_states))
        try:
            yield
        finally:
            self.policy, self.device = learner, device

    def train(self):
        if not os.environ.get("PROFILE_LEARNING"):
            # How many learning steps ran before the KL stop, and what each cost: tells a slower card from more steps.
            policy, calls = self.policy, [0]
            evaluate = policy.evaluate_actions
            kick = self.rollout_buffer.kick
            pull = kick.weight(self.num_timesteps) if kick is not None else 0.0
            dists, pulls = [], []
            if pull > 0:
                to_dist = policy._get_action_dist_from_latent
                policy._get_action_dist_from_latent = lambda latent: dists.append(to_dist(latent)) or dists[-1]

            def counted(*args, **kwargs):
                calls[0] += 1
                values, log_prob, entropy = evaluate(*args, **kwargs)
                if getattr(policy, "hl_gauss", False):
                    # The library's value loss is mean((returns - values)^2). Handing it returns - sqrt(CE) makes
                    # that exactly the cross-entropy of the value's bins against the HL-Gauss target, gradient and
                    # all, without copying its training loop (spatial.HLGaussValue).
                    head = policy.value_net
                    returns = self.rollout_buffer.last_returns
                    logp = torch.log_softmax(head.last_logits, dim=-1)
                    ce = -(head.target(returns) * logp).sum(-1)
                    values = (returns - torch.sqrt(ce + 1e-12)).view_as(values)
                if pull > 0:
                    # The library adds ent_coef * -mean(entropy) to the loss. Handing it entropy - pull/ent_coef * CE
                    # adds exactly pull * mean(CE): the teacher's pull, without copying its training loop.
                    ce = kick.cross_entropy(self.rollout_buffer.last_teacher, dists.pop())
                    real = torch.isfinite(self.rollout_buffer.last_teacher[:, 0])
                    pulls.append(float(ce.detach()[real].mean()))
                    entropy = entropy - (pull / self.ent_coef) * ce.view_as(entropy)
                return values, log_prob, entropy

            policy.evaluate_actions = counted
            if self.device.type == "cuda":
                torch.cuda.empty_cache()  # what playing kept in the cache goes back, so the convolutions have room
            started = time.perf_counter()
            try:
                super().train()
            finally:
                del policy.evaluate_actions
                if pull > 0:
                    del policy._get_action_dist_from_latent
            if pulls:
                self.logger.record("train/kick_weight", pull)
                self.logger.record("train/kick_ce", float(np.mean(pulls)))  # how far from the teacher, in nats
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                self.logger.record("train/free_card_mb", torch.cuda.mem_get_info()[0] // 2**20)
            self.logger.record("train/steps", calls[0])
            self.logger.record("train/ms_per_step", 1000 * (time.perf_counter() - started) / max(calls[0], 1))
            return
        # PROFILE_LEARNING=<file>: the third learning pass under the torch profiler, card and processor time per kernel.
        self._profiled = getattr(self, "_profiled", 0) + 1
        if self._profiled < 3:
            return super().train()
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            super().train()
            torch.cuda.synchronize()
        with open(os.environ["PROFILE_LEARNING"], "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25, max_name_column_width=70))
        os.environ.pop("PROFILE_LEARNING")

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        with self.playing():
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)


def bonus_weight_from(rated, references):
    """How much a fight's event bonuses (long hit, critical hit, missed swing) should count, from a rating.

    They are a crutch for finding those plays; what counts is damage and winning (docs/training-design.md#nedtrappning).
    So they fade as the model gets stronger: full while it scores 0.5 against the frozen copies of itself, nothing at
    0.9. Only the frozen copies count: the scripted bots are already beaten ~90% and measure nothing of these plays.
    The 0.5 and 0.4 are ours, not from a paper.
    """
    score = snapshot_score(rated, references)
    return 1.0 if score is None else weight_for(score)


def snapshot_score(rated, references):
    """The model's average score against the frozen copies of itself this rating fought, or None. A copy made from
    the model in this very rating (the anchor's first, rated["fresh"]) is left out: against itself it scores about
    0.5 whatever it is, which says nothing; one added right after the rating (rating.ADD_AT) has no score yet."""
    fresh = set(rated.get("fresh", ()))
    snapshots = [r["name"] for r in references if r["kind"] == "snapshot" and r["name"] in rated["scores"] and r["name"] not in fresh]
    if not snapshots:
        return None
    return sum(rated["scores"][n] for n in snapshots) / len(snapshots)


def weight_for(score):
    """Full at 0.5 against its frozen copies, nothing at 0.9 (ours, not from a paper)."""
    return float(min(max(1.0 - (score - 0.5) / 0.4, 0.0), 1.0))


# The bonus weight follows the average of this many ratings, not one: with 32 rounds a score is about +-0.09, the weight
# +-0.22, and since it only falls it followed the lowest of a noisy series (auto7 went from 1.0 to 0.22 on one rating).
WEIGHT_OVER = 3


class Progress(BaseCallback):
    def __init__(self, vec, out, curriculum, learn_threads=6):
        super().__init__()
        self.learn_threads = learn_threads
        self.timings = []
        self.vec, self.out, self.cur = vec, out, curriculum
        self.t0 = time.time()
        self.first_step = None
        self.rounds = 0
        self.last_log = self.last_save = self.last_text = self.last_exam = 0
        self.resume_last_exam = None  # when the run's last exam was, if it is continuing (train_sim)
        self.value_warmup = 0  # until this many decisions only the value learns (a new critic memory: freeze_all_but_value)
        # Until this many decisions the learning rate is 0 (STILL_AFTER_SURGERY): after the weights were reshaped the
        # optimizer's moment estimates catch up, and the rounds reach a steady mix, before anything moves. OpenAI Five
        # (arXiv 1912.06680, appendix B) did the same after every surgery.
        self.still_until = 0
        # attr -> [full value, first decision, over how many]: a new rule ramps in from 0 to full (phase_in).
        self.phase_ins = {}
        self.rate_every = 0  # decisions between ratings (train_sim: every other exam, a rating now takes minutes)
        self.last_rated = None
        self.bench_every = None  # decisions between fixed benchmarks (benchmark.py), when train_sim sets it
        self.last_bench = None
        self.rating = None  # rating.Rating: fights against frozen references after each exam (train_sim sets it)
        self.rated = None  # its last result: Elo and score per reference
        self.bonus_weight = 1.0  # how much a fight's event bonuses count (weight_for, shaping.rs); only falls
        self.snapshot_scores = []  # the last WEIGHT_OVER ratings' scores against the frozen copies
        self.pool = None  # pool.Pool: older versions to fight (train_sim sets it, with the vec's opponents)
        self.self_play = {}  # opponent ("latest" or a version) -> recent points, for status
        self.clean = {f: deque(maxlen=WINDOW) for f in FAMILIES}
        self.results, self.falls, self.stalls = deque(maxlen=WINDOW), deque(maxlen=WINDOW), deque(maxlen=WINDOW)
        self.damage, self.speeds, self.shift, self.parts = deque(maxlen=WINDOW), deque(maxlen=WINDOW), deque(maxlen=WINDOW), deque(maxlen=WINDOW)
        self.exams = {}
        self.exams_random = {}
        self.pushed_mastered, self.pushed_practice = [], None
        self.bridge_speeds = deque(maxlen=WINDOW)
        # Blocks per second, best clean bridge of at least 8 blocks. Kept across restarts.
        try:
            self.bridge_record = json.loads((out / "status.json").read_text()).get("bridge_record", 0.0)
        except (OSError, ValueError):
            self.bridge_record = 0.0
        self.progress = self._csv("progress.csv", ["steps", "seconds", "stage", "gap", "directions", "heights", "ticks_per_block", "success", "fell", "stalled", "damage",
                                                   "blocks_per_s", "block_use", "unused", "eval_success", "eval_block_use", "episodes", "shift"])
        self.level_log = self._csv("levels.csv", ["steps", *(f"{f}_level" for f in FAMILIES), *(f"{f}_clean" for f in FAMILIES), *(f"{f}_exam" for f in FAMILIES)])

    def _csv(self, name, header):
        path = self.out / name
        if path.exists():
            with open(path) as f:
                first = f.readline().strip()
            if first != ",".join(header):  # new course kinds or columns: keep the old log beside the new one
                n = 1
                while (self.out / f"{path.stem}_before_{n}.csv").exists():
                    n += 1
                path.rename(self.out / f"{path.stem}_before_{n}.csv")
        new = not path.exists()
        f = open(path, "a", newline="")
        w = csv.writer(f)
        if new:
            w.writerow(header)
        return f, w

    @staticmethod
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    def status(self):
        families = {}
        for f, dials in FAMILIES.items():
            rows = {}
            for d, values in dials.items():
                level = self.cur.levels[f][d]
                rounds = self.cur.limit_rounds[f][d]
                rows[d] = dict(level=level, most=len(values) - 1, value=str(values[level]), rate=self.mean(rounds) if rounds else None, rounds=len(rounds))
            families[f] = dict(name=NAMES[f], open=is_open(self.cur.levels, f), done=finished(self.cur.levels, f), level=self.cur.total(f), most=self.cur.most(f),
                               clean=self.mean(self.clean[f]), exam=self.exams.get(f), exam_random=self.exams_random.get(f), all_dials=self.mean(self.cur.all_rounds[f]), dials=rows)
        parts = {}
        for round_parts in self.parts:
            for rule, value in round_parts.items():
                parts[rule] = parts.get(rule, 0.0) + value
        return dict(steps=self.num_timesteps, stage=self.cur.total(), stages=self.cur.most(), families=families, levels=self.cur.levels, practice=self.cur.practice(), rating=self.rated, bonus_weight=self.bonus_weight,
                    self_play={k: dict(score=self.mean(v), rounds=len(v)) for k, v in self.self_play.items()}, pool=len(self.pool.versions) if self.pool else 0,
                    success=self.mean(self.results), fell=self.mean(self.falls), stalled=self.mean(self.stalls), damage=self.mean(self.damage), speed=self.mean(self.speeds),
                    shift=self.mean(self.shift), bridge_speed=self.mean(self.bridge_speeds), bridge_record=self.bridge_record, parts={k: round(v / max(len(self.parts), 1), 3) for k, v in parts.items()}, seconds=time.time() - self.t0)

    def publish(self):
        status = self.status()
        tmp = self.out / "status.tmp"
        tmp.write_text(json.dumps(status))
        os.replace(tmp, self.out / "status.json")
        return status

    def benchmark(self):
        """The fixed benchmark (benchmark.py), added to benchmark.jsonl: the curve that shows real progress."""
        import benchmark

        self.last_bench = self.num_timesteps
        result = benchmark.run(self.model.policy, self.model.device)
        with open(self.out / "benchmark.jsonl", "a") as f:
            f.write(json.dumps(dict(steps=self.num_timesteps, rounds=benchmark.ROUNDS, result=result)) + "\n")
        brief = ", ".join(f"{fam} {' '.join(str(t['sampled']['clean']) for t in tiers.values())}" for fam, tiers in result.items())
        print(f"benchmark at {self.num_timesteps}: {brief}", flush=True)

    def set_bonus_weight(self, w):
        """The weight for every round from now on (env.rs Env.bonus_weight): the simulator's batch, or each game's AutoEnv."""
        self.bonus_weight = w
        if hasattr(self.vec, "batch"):
            self.vec.batch.set_bonus_weight(w)
        else:
            self.vec.set_attr("bonus_weight", w)

    def phase_in(self, attr, full, over):
        """A new rule (a round setting, e.g. AutoEnv.random_hand) ramps from 0 to `full` over `over` decisions from
        now, instead of all at once. Why: switched on at once it changes too much of what the model meets, and it
        relearns far more than the rule itself. The random hand at once took exam scores from 31 to 19 (bridge)
        and 10 (stairs); OpenAI Five saw the same when it handed the model buyback (arXiv 1912.06680, appendix B)."""
        self.phase_ins[attr] = [float(full), int(self.num_timesteps), int(over)]
        self._phase()

    def _phase(self):
        for attr, (full, start, over) in self.phase_ins.items():
            share = min(1.0, max(0.0, (self.num_timesteps - start) / max(over, 1)))
            self.vec.set_attr(attr, full * share)

    def save_latest(self):
        self.model.save(self.out / "latest_tmp")
        os.replace(self.out / "latest_tmp.zip", self.out / "latest.zip")
        tmp = self.out / "levels.tmp"
        tmp.write_text(json.dumps(self.cur.levels))
        os.replace(tmp, self.out / "levels.json")
        # The steps these weights have, and what the curriculum remembers, so a restart picks up exactly here.
        tmp = self.out / "latest.tmp"
        tmp.write_text(json.dumps(dict(steps=self.num_timesteps, curriculum=self.cur.state(), last_exam=self.last_exam, bonus_weight=self.bonus_weight, snapshot_scores=self.snapshot_scores,
                                    last_bench=self.last_bench, last_rated=self.last_rated, value_warmup=self.value_warmup,
                                    still_until=self.still_until, phase_ins=self.phase_ins)))
        os.replace(tmp, self.out / "latest.json")
        if self.pool is not None:
            self.pool.save()  # the versions' scores
        if self.pool is not None and self.pool.due(self.num_timesteps):
            print(f"pool: added {self.pool.add(self.model, self.num_timesteps)}, {len(self.pool.versions)} versions", flush=True)

    def show(self, title="TRAINING"):
        if time.time() - self.last_text < 1.0:
            return
        self.last_text = time.time()
        lines = []
        for f in FAMILIES:
            if not is_open(self.cur.levels, f):
                lines.append([[f"{NAMES[f]} ", sb.DIM], ["locked", sb.DIM]])
                continue
            rate = self.mean(self.clean[f])
            level, most = self.cur.total(f), self.cur.most(f)
            filled = round(8 * level / most)
            lines.append([[f"{NAMES[f]} ", sb.LABEL], ["■" * filled, "aqua"], ["■" * (8 - filled), sb.DIM], [f" {level}/{most} ", sb.VALUE],
                          [f"{rate:.0%}", sb.higher_is_better(rate, 0.7, 0.4)]])
        lines += [sb.BLANK, sb.stat("Died", f"{self.mean(self.falls):.0%}", sb.lower_is_better(self.mean(self.falls), 0.2, 0.5)),
                  sb.stat("Steps", sb.compact(self.num_timesteps), sb.VALUE)]
        self.vec.env_method("sidebar", [["MINECRAFT RL AGENT ", "white", True], [title, "yellow" if title == "TESTING" else "aqua", True]], lines, indices=0)

    def _on_step(self):
        if self.first_step is None:
            # A resumed run counts from where it is: no save the moment it starts, and the exams keep their schedule
            # (resume_last_exam, from latest.json), so restarts do not keep putting them off.
            self.first_step = self.last_save = self.num_timesteps
            self.last_exam = self.resume_last_exam if self.resume_last_exam is not None else self.num_timesteps
            self.set_bonus_weight(self.bonus_weight)  # a resumed run's weight (train_sim sets it from latest.json)
        moved = False
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.rounds += 1
            family = info["family"]
            if info.get("opponent") is not None:
                # Self-play: its result goes to the pool (how hard each version is now), not to the curriculum or the
                # family's rates, which measure the scripted opponents (docs/training-design.md#self-play).
                points = 1.0 if info["success"] else 0.0 if info["deaths"] > 0 else 0.5
                kind = "latest" if info["opponent"] == "latest" else "pool"
                one = info.get("foes", 1) == 1
                # Against two (the model-played one and a bot) the model loses most rounds whoever the first one is:
                # those say little about how hard a version is, so only one-against-one rounds count for the pool.
                self.self_play.setdefault(kind if one else kind + " 1v2", deque(maxlen=WINDOW)).append(points)
                if self.pool is not None and kind == "pool" and one:
                    self.pool.record(info["opponent"], points)
                self.parts.append(info["parts"])
                continue
            self.clean[family].append(bool(info["clean"]))
            self.results.append(bool(info["clean"]))
            self.falls.append(bool(info["fell"]))
            self.stalls.append(bool(info["stalled"]))
            self.damage.append(float(info["damage"]))
            self.shift.append(float(info["shift_share"]))
            self.parts.append(info["parts"])
            if info["success"] and family != "duel" and "blocks_per_s" in info:  # a won fight has no speed over the course
                self.speeds.append(info["blocks_per_s"])
            if family == "bridge" and info["clean"] and info["blocks_needed"] >= 8:
                self.bridge_speeds.append(info["blocks_per_s"])
                if info["blocks_per_s"] > self.bridge_record:
                    self.bridge_record = info["blocks_per_s"]
                    if self.cur.levels["bridge"]["faster"]:
                        print(f"bridge record {self.bridge_record:.2f} blocks/s at {self.num_timesteps}", flush=True)
            # A round drawn before a dial moved says nothing about the new level: up to a few hundred are still
            # being played when one moves.
            stale = info.get("at") is not None and list(info["at"]) != self.cur.at(family)
            change = None if stale else self.cur.record(family, info["tested"], bool(info["clean"]))
            if change:
                moved = True
                dial, level = change
                print(f"dial {family}.{dial} -> {level}/{len(FAMILIES[family][dial]) - 1} ({FAMILIES[family][dial][level]}) at {self.num_timesteps}, {self.cur.total()}/{self.cur.most()} in all", flush=True)
        mastered = self.cur.mastered()
        practice = {f: round(w, 1) for f, w in self.cur.practice().items()}
        if moved or mastered != self.pushed_mastered or practice != self.pushed_practice:
            self.pushed_mastered, self.pushed_practice = mastered, practice
            self.vec.env_method("set_levels", self.cur.levels, mastered, practice)
        self.show()
        if self.num_timesteps - self.last_log >= 4000:
            self.last_log = self.num_timesteps
            s = self.publish()
            f, w = self.progress
            w.writerow([s["steps"], round(s["seconds"]), s["stage"], 0, "", 0, 0, round(s["success"], 3), round(s["fell"], 3), round(s["stalled"], 3), round(s["damage"], 2), round(s["speed"], 2),
                        0, 0, 0, 0, self.rounds, round(s["shift"], 3)])
            f.flush()
            f, w = self.level_log
            w.writerow([s["steps"], *(self.cur.total(x) for x in FAMILIES), *(round(self.mean(self.clean[x]), 3) for x in FAMILIES), *(self.exams.get(x, "") for x in FAMILIES)])
            f.flush()
            speed = (self.num_timesteps - self.first_step) / max(s["seconds"], 1)
            print(self.num_timesteps, f"level {s['stage']}/{s['stages']}", *(f"{x} {self.cur.total(x)} {self.mean(self.clean[x]):.2f}" for x in FAMILIES if is_open(self.cur.levels, x)),
                  f"died {s['fell']:.2f} stuck {s['stalled']:.2f} {speed:.0f} decisions/s", flush=True)
        if self.num_timesteps - self.last_save >= SAVE_EVERY:
            self.last_save = self.num_timesteps
            self.save_latest()
        return True  # also after the last level: the goal is every round, not just the dials at their limit

    def _on_rollout_start(self):
        now = time.time()
        if getattr(self, "rollout_ended", None):
            self.timings.append(("learn", now - self.rollout_ended))
            if len(self.timings) >= 6:
                collect = np.mean([t for k, t in self.timings if k == "collect"])
                learn = np.mean([t for k, t in self.timings if k == "learn"])
                judged = self.model.logger.name_to_value.get("train/explained_variance", float("nan"))
                print(f"time per stretch of {self.model.n_steps * self.vec.num_envs} decisions: playing {collect:.2f} s, learning {learn:.2f} s, "
                      f"how well it judges what is coming {judged:.2f} (1 is perfect, 0 no better than guessing)", flush=True)
                logged = self.model.logger.name_to_value
                print("learning numbers:", {k.split("/")[1]: round(float(v), 4) for k, v in logged.items() if k.startswith("train/")}, flush=True)
                self.timings = []
        self.rollout_started = now
        torch.set_num_threads(1)  # one action at a time is too small to share out, and spare threads only spin
        share = min(1.0, max(0.0, (self.num_timesteps - CALM_FROM) / CALM_OVER))
        rate = LEARNING_RATE * (LEARNING_RATE_END / LEARNING_RATE) ** share
        if self.num_timesteps < self.still_until:
            rate = 0.0  # Adam still sees the gradients, so its moments are ready when learning starts
        self.model.lr_schedule = get_schedule_fn(rate)
        self._phase()
        self.model.ent_coef = ENTROPY * (ENTROPY_END / ENTROPY) ** share

    def _on_rollout_end(self):
        self.rollout_ended = time.time()
        freeze_all_but_value(self.model.policy, self.num_timesteps < self.value_warmup)
        self.timings.append(("collect", self.rollout_ended - self.rollout_started))
        torch.set_num_threads(self.learn_threads)  # learning on big batches does use them, and the games wait anyway
        if self.num_timesteps - self.last_exam < EXAM_EVERY:
            return
        self.last_exam = self.num_timesteps
        playing = self.vec.opponents if hasattr(self.vec, "opponents") else None
        if playing is not None:
            self.vec.opponents = None  # exams are against the scripted opponents only: they measure the curriculum
        for family in FAMILIES:
            if not is_open(self.cur.levels, family):
                continue
            self.vec.env_method("set_exam", family)
            wins, games = [0], [0]

            def count(local, _globals):
                self.show("TESTING")
                if local["done"]:  # SB3 calls this once per game, with that game's values
                    games[0] += 1
                    wins[0] += bool(local["info"]["clean"])

            evaluate_policy(self.model, self.vec, n_eval_episodes=EXAM_ROUNDS, deterministic=True, callback=count, warn=False)
            self.exams[family] = round(wins[0] / max(games[0], 1), 3)
            print(f"exam at {self.num_timesteps}: {family} {wins[0]}/{games[0]} at level {self.cur.total(family)}/{self.cur.most(family)}", flush=True)
            # The same with the randomness it trains with, to see when the best choice alone falls apart.
            wins[0] = games[0] = 0
            evaluate_policy(self.model, self.vec, n_eval_episodes=EXAM_ROUNDS, deterministic=False, callback=count, warn=False)
            self.exams_random[family] = round(wins[0] / max(games[0], 1), 3)
            print(f"exam with randomness at {self.num_timesteps}: {family} {wins[0]}/{games[0]}", flush=True)
        self.vec.env_method("set_exam", None)
        if playing is not None:
            self.vec.opponents = playing
        if self.bench_every and (self.last_bench is None or self.num_timesteps - self.last_bench >= self.bench_every):
            self.benchmark()
        rate_due = self.last_rated is None or self.num_timesteps - self.last_rated >= self.rate_every
        if self.rating is not None and rate_due:  # strength against frozen references (docs/training-design.md#referenser)
            self.last_rated = self.num_timesteps
            self.rated = self.rating.rate(self.model, self.num_timesteps)
            print(f"rating at {self.num_timesteps}: Elo {self.rated['elo']} +-{self.rated.get('ci', 0)}, "
                  + ", ".join(f"{k} {v:.2f}" for k, v in self.rated["scores"].items()), flush=True)
            if self.rated.get("added"):
                print(f"rating: {self.rated['added']} added to the references at Elo {self.rated['elo']}", flush=True)
            score = snapshot_score(self.rated, self.rating.references)
            if score is not None:
                self.snapshot_scores = (self.snapshot_scores + [score])[-WEIGHT_OVER:]
            enough = len(self.snapshot_scores) == WEIGHT_OVER
            w = weight_for(sum(self.snapshot_scores) / len(self.snapshot_scores)) if enough else 1.0
            if w < self.bonus_weight:  # only falls: a bad rating does not bring the bonuses back
                self.set_bonus_weight(w)
                print(f"event bonus weight now {w:.2f}", flush=True)
        # The test reset the games, so training has to continue from fresh observations.
        self.model._last_obs = self.vec.reset()
        self.model._last_episode_starts = np.ones(self.vec.num_envs, dtype=bool)
        self.publish()


def _whole_rounds(policy, rounds, device):
    """Runs a few whole rounds through the model from their first tick, memory and all, the way it plays them.
    Shorter rounds are padded at the end, which the memory never looks back on. Returns the action distribution and
    which ticks are real."""
    longest = max(len(o) for o, _ in rounds)
    obs = np.zeros((len(rounds), longest, rounds[0][0].shape[1]), np.float32)  # zeros, not NaN: get_distribution goes past MemoryPolicy.extract_features
    actions = np.zeros((len(rounds), longest, rounds[0][1].shape[1]), np.int64)
    real = np.zeros((len(rounds), longest), np.float32)
    for i, (o, act) in enumerate(rounds):
        obs[i, : len(o)], actions[i, : len(o)], real[i, : len(o)] = o, act, 1.0
    obs = torch.as_tensor(obs, device=device).flatten(0, 1)  # round by round, as the memory expects them
    actions = torch.as_tensor(actions, device=device).flatten(0, 1)
    lstm = policy.lstm_actor
    fresh = torch.zeros(lstm.num_layers, len(rounds), lstm.hidden_size, device=device)
    starts = torch.zeros(len(obs), device=device)  # the fresh memory already is the start of each round
    distribution, _ = policy.get_distribution(obs, (fresh, fresh), starts)
    return distribution, actions, torch.as_tensor(real, device=device).flatten()


def _groups(rounds, order, ticks):
    """Splits rounds into batches that stay within a number of ticks once padded to their longest round, so a few
    long rounds do not fill the graphics card."""
    group, longest = [], 0
    for i in order:
        if group and (len(group) + 1) * max(longest, len(rounds[i][0])) > ticks:
            yield group
            group, longest = [], 0
        group.append(rounds[i])
        longest = max(longest, len(rounds[i][0]))
    if group:
        yield group


def start_from_recordings(model, steps, folders, ticks=1024):
    """A new model first learns to play like the recordings, then learns by itself from there. With a memory it
    copies whole rounds, so it also learns what to keep in mind from one tick to the next."""
    files = sorted(path for folder in folders for path in Path(folder).glob("**/round*.jsonl.gz"))
    np.random.default_rng(0).shuffle(files)
    held = max(1, len(files) // 10)
    rounds = [demo_data.replay(path)[:2] for path in files]
    rounds = [(o[:ticks].astype(np.float32), np.asarray(act[:ticks], np.int64)) for o, act in rounds if len(o)]
    val, train = rounds[:held], rounds[held:]
    print(f"recordings: {len(files)} rounds, {sum(len(o) for o, _ in train)} ticks to learn from, {sum(len(o) for o, _ in val)} held back", flush=True)
    device, policy = model.device, model.policy
    params = [p for name, p in policy.named_parameters() if "value" not in name and "vf_" not in name and "critic" not in name]
    opt = torch.optim.Adam(params, lr=1e-3)
    rng = np.random.default_rng(1)
    best, best_state = float("inf"), None
    batches = iter(())
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = 1e-3 * (0.1 ** (step / steps))
        batch = next(batches, None)
        if batch is None:
            batches = _groups(train, rng.permutation(len(train)), ticks)
            batch = next(batches)
        distribution, actions, real = _whole_rounds(policy, batch, device)
        loss = -(distribution.log_prob(actions) * real).sum() / real.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            total, seen, same = 0.0, 0.0, 0.0
            with torch.no_grad():
                for batch in _groups(val, sorted(range(len(val)), key=lambda i: len(val[i][0])), 2 * ticks):
                    distribution, actions, real = _whole_rounds(policy, batch, device)
                    total += -(distribution.log_prob(actions) * real).sum().item()
                    seen += real.sum().item()
                    chosen = torch.stack([d.probs.argmax(1) for d in distribution.distribution], 1)
                    same = same + ((chosen == actions).float() * real[:, None]).sum(0)
            held_loss = total / seen
            if held_loss < best:  # it stops copying details that only fit the rounds it has seen
                best, best_state = held_loss, {k: v.clone() for k, v in policy.state_dict().items()}
            match = (same / seen).cpu().numpy().round(2).tolist()
            print(f"copying step {step}: loss {loss.item():.3f}, on held back {held_loss:.3f}, matches {match}", flush=True)
    policy.load_state_dict(best_state)
    print(f"copying: kept the model that did best on held back rounds, {best:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ports", type=int, nargs="+", default=[25602, 25603, 25606, 25607])
    p.add_argument("--steps", type=int, default=30_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="auto5")
    p.add_argument("--resume", help="model to continue from")
    p.add_argument("--recordings", nargs="+", default=[str(ROOT / "runs" / "demos")], help="folders of recorded rounds to copy first")
    p.add_argument("--start-from-recordings", type=int, default=0, help="steps of copying the recordings before a new model starts")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="where it learns")
    p.add_argument("--epochs", type=int, default=8, help="passes over each stretch of play when learning. More learns more per game tick, but the games wait")
    p.add_argument("--threads", type=int, default=2, help="for learning between rollouts: more measured slower here, 6 took 6.5 s and 2 took 5.5 s")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.backends.cudnn.benchmark = True  # the eyes always see 512 played ticks when learning, so the fastest way is found once
    Distribution.set_default_validate_args(False)  # the checks take a large part of each decision
    out = ROOT / "runs" / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "levels.json"
    curriculum = Curriculum(json.loads(saved.read_text()) if saved.exists() else None)
    vec = SubprocVecEnv([make_env(port, curriculum.levels) for port in a.ports])
    curriculum.load_state((json.loads((out / "latest.json").read_text()) if (out / "latest.json").exists() else {}).get("curriculum", {}))
    model = SplitPPO(MemoryPolicy, vec, seed=a.seed, n_steps=4096 // len(a.ports), batch_size=512, n_epochs=a.epochs, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                     learning_rate=LEARNING_RATE, ent_coef=ENTROPY, device=a.device, target_kl=TARGET_KL, policy_kwargs=POLICY)
    if a.resume:
        grew, model.num_timesteps = load_weights(model, a.resume)
        print(f"continuing from {a.resume}: grown {grew}", flush=True)
    else:
        init_action_bias(model.policy)
        if a.start_from_recordings:
            start_from_recordings(model, a.start_from_recordings, a.recordings)
            model.save(out / "copied_recordings")
    fast_convolutions()
    model.learn(a.steps, callback=Progress(vec, out, curriculum, a.threads), reset_num_timesteps=not a.resume)
    model.save(out / "final")
