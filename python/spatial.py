"""The model's eyes: small networks for what lies around it, next to one for its body, before the usual layers.

A grid of blocks read as a long list of numbers loses what is next to what. These keep it.
"""
import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from auto_env import CUBE_SIZE, CUBE_START, HISTORY_START, MAP_CHANNELS, MAP_SIDE, MAP_START, RADIUS

ENEMIES, ENEMY_F = 3, 20  # duel.rs: the three nearest opponents, 20 values each, first in the fight part
HOTBAR_SLOTS, HOTBAR_KIND_F = 9, 40  # items.rs: per slot its kind (39) and count, then the selected slot (9), last


class SpatialExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, per_part: int = 128):
        super().__init__(observation_space, features_dim=per_part * 3)
        side = 2 * RADIUS + 1
        self.cube_side, self.map_side = side, MAP_SIDE
        self.cube = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.map = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cube_out = self.cube(torch.zeros(1, 1, side, side, side)).shape[1]
            map_out = self.map(torch.zeros(1, MAP_CHANNELS, MAP_SIDE, MAP_SIDE)).shape[1]
        body = CUBE_START + observation_space.shape[0] - HISTORY_START
        self.cube_out = nn.Sequential(nn.Linear(cube_out, per_part), nn.ReLU())
        self.map_out = nn.Sequential(nn.Linear(map_out, per_part), nn.ReLU())
        self.body = nn.Sequential(nn.Linear(body, per_part), nn.ReLU())

    def forward(self, obs):
        n = obs.shape[0]
        cube = obs[:, CUBE_START : CUBE_START + CUBE_SIZE].reshape(n, 1, self.cube_side, self.cube_side, self.cube_side)
        grid = obs[:, MAP_START : HISTORY_START].reshape(n, MAP_CHANNELS, self.map_side, self.map_side)
        body = torch.cat([obs[:, :CUBE_START], obs[:, HISTORY_START:]], dim=1)
        return torch.cat([self.cube_out(self.cube(cube)), self.map_out(self.map(grid)), self.body(body)], dim=1)


class Residual(nn.Module):
    """One step of thinking that adds to what is already there instead of replacing it. The norm keeps the numbers
    in the same range however deep it goes, which is what lets a network like this grow and still learn steadily."""

    def __init__(self, width: int, inner: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, inner), nn.SiLU(), nn.Linear(inner, width))

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class SpatialNet(BaseFeaturesExtractor):
    """The eyes of the newer models. The first layer of each grid looks at every block before anything is merged,
    so a single block, or which side of an edge it stands on, is still seen exactly. The parts then meet in a few
    residual steps. What it remembers between ticks is added after this, in MemoryPolicy."""

    def __init__(self, observation_space: gym.spaces.Box, per_part: int = 128, width: int = 256, depth: int = 2, entities: bool = False):
        super().__init__(observation_space, features_dim=width)
        side = 2 * RADIUS + 1
        self.cube_side, self.map_side = side, MAP_SIDE
        self.cube = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1), nn.SiLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
        )
        self.map = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, 3, stride=1, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cube_out = self.cube(torch.zeros(1, 1, side, side, side)).shape[1]
            map_out = self.map(torch.zeros(1, MAP_CHANNELS, MAP_SIDE, MAP_SIDE)).shape[1]
        body = CUBE_START + observation_space.shape[0] - HISTORY_START
        self.cube_out = nn.Sequential(nn.Linear(cube_out, per_part), nn.LayerNorm(per_part))
        self.map_out = nn.Sequential(nn.Linear(map_out, per_part), nn.LayerNorm(per_part))
        self.body = nn.Sequential(nn.Linear(body, per_part), nn.LayerNorm(per_part))
        # What the body part sees, standardized: (x - mean) / std. Identity until train_sim measures the inputs
        # (--normalize-inputs) and folds the measured values into the first layer, so the model answers as before.
        self.register_buffer("body_mean", torch.zeros(body))
        self.register_buffer("body_std", torch.ones(body))
        self.join = nn.Linear(3 * per_part, width)
        self.think = nn.Sequential(*(Residual(width, 2 * width) for _ in range(depth)), nn.LayerNorm(width))
        # Opponents and hotbar slots each through one small shared network, then pooled: the same thing learnt about
        # one opponent or slot holds for every one, whatever the order or the count (OpenAI Five, AlphaStar). Added to
        # the body part through a layer that starts at zero, so a model moved over answers exactly as before.
        self.entities = entities
        if entities:
            self.enemy = nn.Sequential(nn.Linear(ENEMY_F, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
            self.slot = nn.Sequential(nn.Linear(HOTBAR_KIND_F, 32), nn.SiLU())
            self.entity_out = nn.Linear(64 + 2 * 32, per_part)
            nn.init.zeros_(self.entity_out.weight)
            nn.init.zeros_(self.entity_out.bias)

    def forward(self, obs):
        n = obs.shape[0]
        cube = obs[:, CUBE_START : CUBE_START + CUBE_SIZE].reshape(n, 1, self.cube_side, self.cube_side, self.cube_side)
        grid = obs[:, MAP_START : HISTORY_START].reshape(n, MAP_CHANNELS, self.map_side, self.map_side)
        body = (torch.cat([obs[:, :CUBE_START], obs[:, HISTORY_START:]], dim=1) - self.body_mean) / self.body_std
        lin = self.body[0](body)
        if self.entities:
            lin = lin + self.entity_out(self._entities(obs))
        parts = torch.cat([self.cube_out(self.cube(cube)), self.map_out(self.map(grid)), self.body[1](lin)], dim=1)
        return self.think(self.join(nn.functional.silu(parts)))

    def _entities(self, obs):
        """Opponents: each through `enemy`, max-pooled over those there (its first value says it is). Hotbar: each
        slot through `slot`, summed, and the selected slot's own."""
        n = obs.shape[0]
        start = obs.shape[1] - (HOTBAR_SLOTS * HOTBAR_KIND_F + HOTBAR_SLOTS)
        extra0 = start - (ENEMIES * ENEMY_F + 9 + 4 + 5)  # duel.rs EXTRA: enemies, self, command, clicks, hotbar
        foes = obs[:, extra0 : extra0 + ENEMIES * ENEMY_F].reshape(n, ENEMIES, ENEMY_F)
        there = foes[:, :, :1] > 0.5
        e = self.enemy(foes)
        e = torch.where(there, e, torch.full_like(e, -1e4)).max(dim=1).values
        e = torch.where(there.any(dim=1), e, torch.zeros_like(e))
        slots = obs[:, start : start + HOTBAR_SLOTS * HOTBAR_KIND_F].reshape(n, HOTBAR_SLOTS, HOTBAR_KIND_F)
        selected = obs[:, start + HOTBAR_SLOTS * HOTBAR_KIND_F :]
        h = self.slot(slots)
        return torch.cat([e, h.sum(dim=1), (h * selected.unsqueeze(2)).sum(dim=1)], dim=1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A model saved before the standardization had no such values: it loads with none (mean 0, std 1).
        state_dict.setdefault(prefix + "body_mean", self.body_mean.clone())
        state_dict.setdefault(prefix + "body_std", self.body_std.clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def fold_normalization(self, mean, std):
        """Standardize the body part with these measured values, and change the first layer so the model answers
        exactly as before: W (x - m) / s + b' = W x + b when the layer's columns become W s and b' = b + W m."""
        lin = self.body[0]
        old_mean, old_std = self.body_mean.clone(), self.body_std.clone()
        # Undo the old standardization in the weights, then apply the new one.
        w_raw = lin.weight / old_std  # weights on raw x
        b_raw = lin.bias - (lin.weight * (old_mean / old_std)).sum(dim=1)
        lin.weight.copy_(w_raw * std)
        lin.bias.copy_(b_raw + (w_raw * mean).sum(dim=1))
        self.body_mean.copy_(mean)
        self.body_std.copy_(std)


from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy  # noqa: E402


from stable_baselines3.common.distributions import MultiCategoricalDistribution  # noqa: E402

# How much each action's randomness counts in the entropy bonus, by action (move, strafe, sneak, use, jump, sprint, yaw,
# pitch, attack, hotbar). The hotbar key counts a tenth: with the full bonus, exploration pushed random slot changes to
# about 1 % of ticks, and every change puts something else in hand; in fights it held the sword only 23 % of the time,
# and on courses to build it held the sword so often that bridging fell from 0.85 to 0.1 (commits a7d4eec, 900f394). A
# change that helps is still learned from the reward; it is just not pushed toward at random. Our own choice, not from a
# paper (docs/training-design.md#foremal).
ENTROPY_WEIGHTS = (1.0,) * 9 + (0.1,)


class WeightedMultiCategorical(MultiCategoricalDistribution):
    """MultiCategoricalDistribution whose entropy weighs each action by ENTROPY_WEIGHTS."""

    def entropy(self):
        e = torch.stack([d.entropy() for d in self.distribution], dim=1)
        w = torch.tensor(ENTROPY_WEIGHTS[: e.shape[1]] + (1.0,) * max(0, e.shape[1] - len(ENTROPY_WEIGHTS)), dtype=e.dtype, device=e.device)
        return (e * w).sum(dim=1)


class HLGaussValue(nn.Module):
    """The value as a distribution over BINS bins of the (scaled) return instead of one number, trained by
    cross-entropy against the return smeared by a Gaussian (HL-Gauss: Farebrother et al. 2024, "Stop Regressing",
    arXiv 2403.03950, ICML): robust to noisy and moving targets, which ours are (seven kinds of round, self-play). Its
    forward gives the expected value, so rollouts and GAE are unchanged; the logits stay for the loss (SplitPPO)."""

    LOW, HIGH, BINS = -4.0, 4.0, 101  # scaled returns: a win is +1, a whole course rarely over 3
    SIGMA = 0.75  # in bin widths, the paper's choice

    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, self.BINS)
        self.register_buffer("edges", torch.linspace(self.LOW, self.HIGH, self.BINS + 1))
        self.last_logits = None

    def forward(self, x):
        self.last_logits = self.linear(x)
        centers = (self.edges[:-1] + self.edges[1:]) / 2
        return (torch.softmax(self.last_logits, dim=-1) * centers).sum(-1, keepdim=True)

    def target(self, returns):
        """The return as probabilities over the bins: a Gaussian around it, integrated over each bin."""
        sigma = self.SIGMA * (self.HIGH - self.LOW) / self.BINS
        r = returns.clamp(self.LOW, self.HIGH).unsqueeze(-1)
        cdf = torch.special.erf((self.edges - r) / (sigma * 2 ** 0.5))
        p = cdf[..., 1:] - cdf[..., :-1]
        return p / p.sum(-1, keepdim=True).clamp_min(1e-12)


class MemoryPolicy(RecurrentActorCriticPolicy):
    """Carries a memory from tick to tick, so it can hold on to a plan and to what just happened.

    The library steps through a stretch one tick at a time whenever a round starts inside it. Stretches here are
    always cut so a round can only start at their first tick, so that reset is done once up front and the whole
    stretch runs in one go, which is many times faster when learning.

    Ticks that only pad a stretch out are marked with NaN as their first value. The eyes skip them and they get
    nothing to see, which keeps the costly part of learning to the ticks that were really played."""

    def __init__(self, *args, hl_gauss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hl_gauss = hl_gauss
        if hl_gauss:
            self.value_net = HLGaussValue(self.mlp_extractor.latent_dim_vf)
            lr = args[2] if len(args) > 2 else kwargs["lr_schedule"]
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr(1), **self.optimizer_kwargs)
        if type(self.action_dist) is MultiCategoricalDistribution:
            self.action_dist.__class__ = WeightedMultiCategorical  # same state, the entropy weighed per action
        # The library initializes every layer again after building (orthogonal); the entity layer has to start at zero.
        for extractor in {id(e): e for e in (self.features_extractor, getattr(self, "pi_features_extractor", None), getattr(self, "vf_features_extractor", None)) if e is not None}.values():
            if getattr(extractor, "entities", False):
                torch.nn.init.zeros_(extractor.entity_out.weight)
                torch.nn.init.zeros_(extractor.entity_out.bias)

    def extract_features(self, obs, features_extractor=None):
        real = ~torch.isnan(obs[:, 0])
        if bool(real.all()):
            return super().extract_features(obs, features_extractor)
        seen = super().extract_features(obs[real], features_extractor)
        out = seen.new_zeros((len(obs), seen.shape[1]))
        out[real] = seen
        return out

    @staticmethod
    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["hl_gauss"] = self.hl_gauss  # so a saved model is rebuilt with its value head
        return data

    @staticmethod
    def _process_sequence(features, lstm_states, episode_starts, lstm):
        n_seq = lstm_states[0].shape[1]
        sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        keep = (1.0 - starts[0]).view(1, n_seq, 1)
        output, lstm_states = lstm(sequence, (keep * lstm_states[0], keep * lstm_states[1]))
        return torch.flatten(output.transpose(0, 1), start_dim=0, end_dim=1), lstm_states
