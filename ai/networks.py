"""Hunter policy network: a CNN over the multi-channel map image plus an MLP
over scalar features, plugged into Stable-Baselines3 as a features extractor."""
from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class NeuroHuntExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_dim: int = 256, vec_dim: int = 64):
        super().__init__(observation_space, features_dim=cnn_dim + vec_dim)
        c, h, w = observation_space["grid"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(n, cnn_dim), nn.ReLU())
        self.vec = nn.Sequential(nn.Linear(observation_space["vec"].shape[0], vec_dim), nn.ReLU())

    def forward(self, obs):
        return torch.cat([self.cnn_head(self.cnn(obs["grid"])), self.vec(obs["vec"])], dim=1)
