"""The model's eyes: small networks for what lies around it, next to one for its body, before the usual layers.

A grid of blocks read as a long list of numbers loses what is next to what. These keep it.
"""
import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from auto_env import CUBE_SIZE, CUBE_START, HISTORY_START, MAP_CHANNELS, MAP_SIDE, MAP_START, RADIUS


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

    def __init__(self, observation_space: gym.spaces.Box, per_part: int = 128, width: int = 256, depth: int = 2):
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
        self.join = nn.Linear(3 * per_part, width)
        self.think = nn.Sequential(*(Residual(width, 2 * width) for _ in range(depth)), nn.LayerNorm(width))

    def forward(self, obs):
        n = obs.shape[0]
        cube = obs[:, CUBE_START : CUBE_START + CUBE_SIZE].reshape(n, 1, self.cube_side, self.cube_side, self.cube_side)
        grid = obs[:, MAP_START : HISTORY_START].reshape(n, MAP_CHANNELS, self.map_side, self.map_side)
        body = torch.cat([obs[:, :CUBE_START], obs[:, HISTORY_START:]], dim=1)
        parts = torch.cat([self.cube_out(self.cube(cube)), self.map_out(self.map(grid)), self.body(body)], dim=1)
        return self.think(self.join(nn.functional.silu(parts)))


from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy  # noqa: E402


class MemoryPolicy(RecurrentActorCriticPolicy):
    """Carries a memory from tick to tick, so it can hold on to a plan and to what just happened.

    The library steps through a stretch one tick at a time whenever a round starts inside it. Stretches here are
    always cut so a round can only start at their first tick, so that reset is done once up front and the whole
    stretch runs in one go, which is many times faster when learning.

    Ticks that only pad a stretch out are marked with NaN as their first value. The eyes skip them and they get
    nothing to see, which keeps the costly part of learning to the ticks that were really played."""

    def extract_features(self, obs, features_extractor=None):
        real = ~torch.isnan(obs[:, 0])
        if bool(real.all()):
            return super().extract_features(obs, features_extractor)
        seen = super().extract_features(obs[real], features_extractor)
        out = seen.new_zeros((len(obs), seen.shape[1]))
        out[real] = seen
        return out

    @staticmethod
    def _process_sequence(features, lstm_states, episode_starts, lstm):
        n_seq = lstm_states[0].shape[1]
        sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        keep = (1.0 - starts[0]).view(1, n_seq, 1)
        output, lstm_states = lstm(sequence, (keep * lstm_states[0], keep * lstm_states[1]))
        return torch.flatten(output.transpose(0, 1), start_dim=0, end_dim=1), lstm_states
