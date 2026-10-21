"""Scripted player strategies the hunter trains against.

Each profile has habits (preferred locker, preferred exit, how cautious it
is, how often it throws distractions, how noisy it moves). Profiles are
randomised per episode so the hunter must generalise instead of memorising a
single opponent. A held-out profile ("ghost") is only used for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LOCKER
from .world import NEIGH, World


@dataclass
class PlayerProfile:
    name: str
    hide_bias: float      # prob. of using the preferred locker when hiding
    exit_bias: float      # prob. of using the preferred exit
    caution: float        # distance at which it reacts to the hunter
    distract_prob: float  # prob. of throwing a distraction when spotted
    run_prob: float       # prob. a move is a noisy run instead of a sneak
    hide_time: tuple[int, int] = (5, 14)
    eps: float = 0.08     # random-move noise


TRAIN_PROFILES = [
    PlayerProfile("wanderer", 0.25, 0.30, 4.0, 0.10, 0.50),
    PlayerProfile("hider", 0.85, 0.50, 7.0, 0.05, 0.20, (8, 20)),
    PlayerProfile("runner", 0.40, 0.90, 3.0, 0.05, 0.85),
    PlayerProfile("trickster", 0.50, 0.60, 6.0, 0.60, 0.30),
    PlayerProfile("creature_of_habit", 0.95, 0.95, 5.0, 0.20, 0.40),
    PlayerProfile("unpredictable", 0.0, 0.0, 5.0, 0.25, 0.45),
]
EVAL_PROFILES = TRAIN_PROFILES + [PlayerProfile("ghost", 0.70, 0.75, 8.0, 0.35, 0.15, (10, 24))]


class ScriptedPlayer:
    def __init__(self, world: World, profile: PlayerProfile, rng: np.random.Generator):
        self.w, self.p, self.rng = world, profile, rng
        m = world.map
        self.pref_locker = m.lockers[int(rng.integers(len(m.lockers)))] if m.lockers else None
        self.pref_exit = m.exits[int(rng.integers(len(m.exits)))]
        self.target_exit = self.pref_exit if rng.random() < profile.exit_bias else m.exits[int(rng.integers(len(m.exits)))]
        self.hide_timer = 0
        self.flee_to = None
        self.throws_left = 3
        self.spotted_before = False
        self.stamina = 10.0      # sprint budget (an extra tile per step while fleeing)

    def sample_history(self, behavior, n: int):
        """Fill the behaviour model with ``n`` synthetic past games of this
        profile - what the hunter would remember from earlier sessions."""
        m = self.w.map
        for _ in range(n):
            if m.lockers and self.rng.random() < 0.8:
                l = self.pref_locker if self.rng.random() < self.p.hide_bias else m.lockers[int(self.rng.integers(len(m.lockers)))]
                behavior.record_hide(l)
            e = self.pref_exit if self.rng.random() < self.p.exit_bias else m.exits[int(self.rng.integers(len(m.exits)))]
            behavior.record_exit(e)
            for _ in range(int(self.rng.poisson(self.p.distract_prob * 2))):
                behavior.record_distraction()
            behavior.moves += 60
            behavior.dir_changes += 60 * (0.2 + 0.5 * self.p.eps + 0.3 * (1 - self.p.exit_bias) * 0.5)
            behavior.end_episode()

    # ------------------------------------------------------------------
    def goal(self):
        w = self.w
        if not w.has_keycard:
            return w.map.keycard
        if not w.terminal_restored:
            return w.map.terminal
        return self.target_exit

    def sees_hunter(self) -> bool:
        w = self.w
        d2 = (w.hunter[0] - w.player[0]) ** 2 + (w.hunter[1] - w.player[1]) ** 2
        return d2 <= self.p.caution ** 2 and w.line_of_sight(w.player, w.hunter)

    def act(self, behavior) -> dict:
        """Take one player step. Returns events for reward/behaviour tracking."""
        w, rng, p = self.w, self.rng, self.p
        ev = {"hid": None, "exit_used": None, "distraction": False, "moved": None}
        threat = self.sees_hunter()

        if w.player_hidden:
            self.hide_timer -= 1
            near = abs(w.hunter[0] - w.player[0]) + abs(w.hunter[1] - w.player[1]) <= 3
            if self.hide_timer <= 0 and not (threat and near):
                w.toggle_hide()
            return ev

        if threat:
            if not self.spotted_before and self.throws_left and rng.random() < p.distract_prob:
                self.throw_distraction()
                ev["distraction"] = True
                behavior.record_distraction()
            self.spotted_before = True
            self.flee_to = self.pick_locker() if self.rng.random() < 0.6 + 0.4 * p.hide_bias or self.flee_to else None
        else:
            self.spotted_before = False

        if (self.flee_to is not None and w.player == self.flee_to and w.grid[w.player[1], w.player[0]] == LOCKER
                and not w.player_visible):  # only duck in once out of the hunter's sight
            w.toggle_hide()
            self.hide_timer = int(rng.integers(*p.hide_time))
            ev["hid"] = w.player
            behavior.record_hide(w.player)
            self.flee_to = None
            return ev
        if self.flee_to is not None and w.player == self.flee_to and w.player_visible:
            self.flee_to = None          # seen at the locker: hiding now would be suicide, keep running

        fleeing = threat or self.flee_to is not None
        if not fleeing and rng.random() > self.w.cfg.player_move_prob:
            return ev                                   # hesitates this step
        target = self.flee_to or self.goal()
        if rng.random() < p.eps and not threat:
            d = NEIGH[int(rng.integers(4))]
        else:
            d = self.choose_step(target, threat)
        moved = d != (0, 0) and w.player_move(d)
        if moved and fleeing and self.stamina >= 1.0:
            self.stamina -= 1.0                         # sprint: one extra tile
            d2 = self.choose_step(target, threat)
            if d2 != (0, 0):
                w.player_move(d2)
        else:
            self.stamina = min(10.0, self.stamina + 0.15)
        if moved:
            ev["moved"] = d
            behavior.record_move(d)
            if fleeing or rng.random() < p.run_prob:
                w.emit_noise(w.player, 1.0, real=True)     # footsteps
            else:
                w.emit_noise(w.player, 0.35, real=True)    # sneaking
        if w.player == w.map.terminal:
            w.restore_terminal()
        if w.escaped():
            ev["exit_used"] = w.player
            behavior.record_exit(w.player)
        return ev

    def pick_locker(self):
        """A locker the player can reach clearly before the hunter does."""
        w = self.w
        safe = [l for l in w.map.lockers if w.path_dist(w.player, l) + 2 <= w.path_dist(w.hunter, l)]
        if not safe:
            return None
        if self.pref_locker in safe and self.rng.random() < self.p.hide_bias:
            return self.pref_locker
        return min(safe, key=lambda l: w.path_dist(w.player, l))

    def choose_step(self, target, threat):
        """Head for ``target`` but keep away from the hunter when it is known."""
        w = self.w
        dist_goal = w.distance_field(target)
        knows = threat or (w.line_of_sight(w.player, w.hunter) and
                           (w.hunter[0] - w.player[0]) ** 2 + (w.hunter[1] - w.player[1]) ** 2 <= 100)
        best, best_d = None, None
        for d in ((0, 0),) + NEIGH:
            n = (w.player[0] + d[0], w.player[1] + d[1])
            if not w.passable(n):
                continue
            score = float(dist_goal[n[1], n[0]])
            if knows:
                hd = w.path_dist(n, w.hunter)
                if threat and self.flee_to is None:
                    score = 0.3 * score - 3.0 * min(hd, 12)   # no safe locker: just get away
                else:
                    score += 4.0 * max(0, 5 - hd)
            score += 0.3 if d == (0, 0) else 0.0
            score += self.rng.random() * 0.2
            if best is None or score < best:
                best, best_d = score, d
        return best_d or (0, 0)

    def throw_distraction(self):
        w, rng = self.w, self.rng
        # throw to a reachable tile roughly away from the player's goal
        cands = [(x, y) for y in range(w.h) for x in range(w.w)
                 if w.passable((x, y)) and 16 <= (x - w.player[0]) ** 2 + (y - w.player[1]) ** 2 <= 49]
        if cands:
            spot = cands[int(rng.integers(len(cands)))]
            w.emit_noise(spot, 1.4, real=False)
            self.throws_left -= 1
