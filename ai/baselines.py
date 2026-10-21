"""Baseline hunters used to judge whether PPO learned anything worthwhile.

All baselines act through the *same* action space and observations as the
PPO hunter, so the comparison is apples to apples.
"""
from __future__ import annotations

import numpy as np

from environment.config import A_CHECK_HIDE, A_INTERCEPT, A_INVESTIGATE, A_PURSUE, A_SCAN


class RandomHunter:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, env=None):
        return int(self.rng.integers(0, 5))   # primitive moves + scan only


class PatrolChaser:
    """Classic game AI: chase what you see, investigate what you hear,
    otherwise patrol random rooms. No memory of the player's habits."""
    name = "patrol_chaser"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.waypoint = None

    def predict(self, obs, env):
        w = env.world
        if w.player_visible or (w.last_known is not None and not w.last_known_investigated):
            return A_PURSUE
        if w.loudest_heard() is not None:
            return A_INVESTIGATE
        if self.waypoint is None or w.hunter == self.waypoint or self.rng.random() < 0.02:
            rx, ry, rw, rh = w.map.rooms[int(self.rng.integers(len(w.map.rooms)))]
            self.waypoint = (rx + rw // 2, ry + rh // 2)
            if not w.passable(self.waypoint):
                self.waypoint = None
                return A_SCAN
        n = w.step_toward(w.hunter, self.waypoint)
        d = (n[0] - w.hunter[0], n[1] - w.hunter[1])
        return {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}.get(d, A_SCAN)


class HabitHeuristic(PatrolChaser):
    """Hand-written rules that *also* read the behaviour model: after losing
    the player, check their favourite locker; after the alarm, guard their
    favourite exit. The strongest scripted opponent."""
    name = "habit_heuristic"

    def predict(self, obs, env):
        w = env.world
        if w.player_visible or (w.last_known is not None and not w.last_known_investigated):
            return A_PURSUE
        if w.alarm and env.behavior.predicted_exit(w.map.exits) is not None:
            return A_INTERCEPT
        if w.last_known is not None and w.last_known_age < 25 and env.behavior.hide_scores(w.map.lockers):
            return A_CHECK_HIDE
        if w.loudest_heard() is not None:
            return A_INVESTIGATE
        return super().predict(obs, env)


BASELINES = {"random": RandomHunter, "patrol_chaser": PatrolChaser, "habit_heuristic": HabitHeuristic}
