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
