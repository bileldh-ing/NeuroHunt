"""Reward function. Values come from ``RewardConfig`` (the design-doc table)
and are starting points for experimentation, not a proven optimum."""
from __future__ import annotations

from .config import RewardConfig, WALL
from .world import World


class RewardTracker:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.intercept_given = False
        self.consecutive_bumps = 0
        self.investigations = 0
        self.false_investigations = 0
        self.breakdown = {k: 0.0 for k in ("catch", "intercept", "clue", "track", "empty_search", "wall_bump", "escape", "time_step")}

    def _add(self, key: str, value: float) -> float:
        self.breakdown[key] += value
        return value

    def step(self, world: World, hunter_ev: dict, caught: bool, escaped: bool, player_target) -> float:
        c = self.cfg
        r = self._add("time_step", c.time_step)
        if caught:
            r += self._add("catch", c.catch)
        if escaped:
            r += self._add("escape", c.escape)
        if world.player_visible:
            r += self._add("track", c.track)

        # wall bumps: only repeated collisions are punished at full weight
        if hunter_ev["bumped"]:
            self.consecutive_bumps += 1
            r += self._add("wall_bump", c.wall_bump * (0.25 if self.consecutive_bumps == 1 else 1.0))
        else:
            self.consecutive_bumps = 0

        # scanning (or an intent with no target) far from anything useful
        if hunter_ev["scanned"] and not world.player_visible:
            hx, hy = world.hunter
            near_locker = any(world.grid[hy + dy, hx + dx] == 2 for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
                              if 0 <= hx + dx < world.w and 0 <= hy + dy < world.h)
            if not near_locker:
                r += self._add("empty_search", c.empty_search * (1.0 if hunter_ev["no_target"] else 0.2))

        # investigating a useful clue: reaching a real noise or the last known position
        for n in world.noises:
            if not n.investigated and abs(n.pos[0] - world.hunter[0]) + abs(n.pos[1] - world.hunter[1]) <= 1:
                n.investigated = True
                self.investigations += 1
                if not n.real:
                    self.false_investigations += 1
                if n.real and n.age <= 12:
                    r += self._add("clue", c.clue * 0.5)
        if world.last_known is not None and not world.last_known_investigated and not world.player_visible:
            if abs(world.last_known[0] - world.hunter[0]) + abs(world.last_known[1] - world.hunter[1]) <= 1:
                world.last_known_investigated = True
                r += self._add("clue", c.clue)

        # intercept: hunter guards the exit the player is heading for before it gets there
        if not self.intercept_given and world.has_keycard and player_target is not None:
            if world.path_dist(world.hunter, player_target) <= 2 and world.path_dist(world.player, player_target) <= 10:
                self.intercept_given = True
                r += self._add("intercept", c.intercept)
        return r * c.scale
