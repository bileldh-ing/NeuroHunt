"""Gymnasium training environment for the NeuroHunt hunter.

Headless and fast: no rendering, thousands of short episodes per minute.
Each episode samples a new procedural map (or a fixed JSON map), a scripted
player profile and a synthetic play history for the behaviour model.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai.player_behavior import PlayerBehaviorModel

from . import maps, observations
from .config import N_ACTIONS, EnvConfig
from .player_bots import EVAL_PROFILES, TRAIN_PROFILES, PlayerProfile, ScriptedPlayer
from .rewards import RewardTracker
from .world import World


class NeuroHuntEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, cfg: EnvConfig | None = None, eval_mode: bool = False,
                 profiles: list[PlayerProfile] | None = None, map_path: str | None = None,
                 seed_offset: int = 0):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.eval_mode = eval_mode
        self.profiles = profiles or (EVAL_PROFILES if eval_mode else TRAIN_PROFILES)
        self.map_path = map_path
        self.seed_offset = seed_offset + (maps.EVAL_SEED_OFFSET if eval_mode else 0)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = observations.observation_space(15, 21)
        self.rewards = RewardTracker(self.cfg.rewards)
        self._episode = 0
        self.world: World | None = None

    def _new_map(self):
        if self.map_path:
            return maps.load_json(self.map_path)
        if self.eval_mode:   # held-out maps: deterministic, never seen in training
            seed = self.seed_offset + int(self.np_random.integers(0, 10_000))
        else:
            seed = int(self.np_random.integers(0, maps.EVAL_SEED_OFFSET))
        return maps.generate(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self._episode += 1
        mapdata = options.get("map") or self._new_map()
        self.world = World(mapdata, self.cfg)
        self.profile = options.get("profile") or self.profiles[int(self.np_random.integers(len(self.profiles)))]
        self.player = ScriptedPlayer(self.world, self.profile, self.np_random)
        self.behavior = PlayerBehaviorModel()
        n_hist = options.get("history", int(self.np_random.integers(0, self.cfg.history_episodes_max + 1)))
        self.player.sample_history(self.behavior, n_hist)
        self.rewards.reset()
        self.steps = 0
        return observations.build(self.world, self.behavior, 0, self.cfg.max_steps), {}

    def step(self, action):
        w = self.world
        w.t = self.steps
        hev = w.hunter_act(int(action), self.behavior, self.np_random)
        w.update_perception()
        caught = w.caught()
        escaped = False
        if not caught:
            self.player.act(self.behavior)
            w.update_perception()
            caught = w.caught()
            escaped = (not caught) and w.escaped()
        w.age_noises()
        self.steps += 1
        target = self.player.target_exit if w.terminal_restored else None
        reward = self.rewards.step(w, hev, caught, escaped, target)
        terminated = caught or escaped
        truncated = (not terminated) and self.steps >= self.cfg.max_steps
        info = {}
        if terminated or truncated:
            outcome = "caught" if caught else ("escaped" if escaped else "timeout")
            info = {"outcome": outcome, "steps": self.steps, "profile": self.profile.name,
                    "reward_breakdown": dict(self.rewards.breakdown), 
                    "investigations": self.rewards.investigations,
                    "false_investigations": self.rewards.false_investigations}
        obs = observations.build(w, self.behavior, self.steps, self.cfg.max_steps)
        return obs, reward, terminated, truncated, info

    def render(self):
        w = self.world
        chars = {0: ".", 1: "#", 2: "L", 3: "E", 4: "K", 5: "T"}
        rows = []
        for y in range(w.h):
            row = ""
            for x in range(w.w):
                if (x, y) == w.hunter:
                    row += "H"
                elif (x, y) == w.player:
                    row += "p" if w.player_hidden else "P"
                else:
                    row += chars[int(w.grid[y, x])]
            rows.append(row)
        return "\n".join(rows)


def make_env(cfg: EnvConfig | None = None, eval_mode: bool = False, rank: int = 0, seed: int = 0, **kw):
    def _init():
        env = NeuroHuntEnv(cfg, eval_mode=eval_mode, **kw)
        env.reset(seed=seed + rank)
        return env
    return _init
