"""Shared constants and configuration for the NeuroHunt simulation.

The same simulation core (``environment.world``) is used by the headless
training environment and by the playable Pygame game, so every rule defined
here applies identically to both.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --- Tile types -------------------------------------------------------------
FLOOR = 0
WALL = 1
LOCKER = 2      # hiding spot (player can hide inside)
EXIT = 3        # escape hatch (needs keycard + restored terminal)
KEYCARD = 4     # keycard pickup
TERMINAL = 5    # terminal the player must restore

TILE_CHARS = {
    "#": WALL, ".": FLOOR, "L": LOCKER, "E": EXIT, "K": KEYCARD, "T": TERMINAL,
    "P": FLOOR, "H": FLOOR,  # player / hunter spawn markers sit on floor
}

MAP_W = 21
MAP_H = 15

# --- Hunter action space ----------------------------------------------------
# 0-3 primitive moves, 4 scan, 5-8 macro "intent" actions that take one
# path-finding step toward a target the hunter itself chose to pursue.
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_SCAN, A_PURSUE, A_INVESTIGATE, A_INTERCEPT, A_CHECK_HIDE = range(9)
N_ACTIONS = 9
ACTION_NAMES = [
    "move_up", "move_down", "move_left", "move_right", "scan",
    "pursue_last_known", "investigate_noise", "intercept_exit", "check_hiding_spot",
]
DIRS = {A_UP: (0, -1), A_DOWN: (0, 1), A_LEFT: (-1, 0), A_RIGHT: (1, 0)}


@dataclass
class RewardConfig:
    """Starting reward values from the design doc. ``scale`` keeps PPO's
    value targets in a sane range without changing their ratios."""
    catch: float = 100.0
    intercept: float = 30.0
    clue: float = 10.0
    track: float = 2.0
    empty_search: float = -5.0
    wall_bump: float = -10.0
    escape: float = -100.0
    time_step: float = -0.1
    scale: float = 0.01


@dataclass
class EnvConfig:
    max_steps: int = 300
    vision_range: float = 6.0
    fov_deg: float = 100.0
    scan_range: float = 4.5
    hearing_range: float = 6.0
    player_move_prob: float = 0.85     # player is slightly slower than the hunter
    history_episodes_max: int = 8      # synthetic past games fed to the behavior model
    map_pool: str = "procedural"        # or a path to a JSON map
    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "EnvConfig":
        data = json.loads(Path(path).read_text())
        env = data.get("env", data)
        rewards = RewardConfig(**env.pop("rewards", {}))
        known = {k: v for k, v in env.items() if k in cls.__dataclass_fields__}
        return cls(rewards=rewards, **known)

    def to_dict(self) -> dict:
        return asdict(self)
