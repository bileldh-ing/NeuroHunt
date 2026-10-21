"""Simulation core shared by the training environment and the Pygame game.

The hunter has *limited* perception: a vision cone blocked by walls, a short
360-degree scan, and hearing for noises within range. It never receives the
player's true position unless it actually perceives it.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .config import (A_CHECK_HIDE, A_INTERCEPT, A_INVESTIGATE, A_PURSUE, A_SCAN, DIRS,
                     EXIT, KEYCARD, LOCKER, WALL, EnvConfig)
from .maps import MapData

Pos = tuple[int, int]
NEIGH = ((0, -1), (0, 1), (-1, 0), (1, 0))


class Noise:
    __slots__ = ("pos", "intensity", "age", "real", "investigated")

    def __init__(self, pos: Pos, intensity: float, real: bool):
        self.pos, self.intensity, self.age, self.real, self.investigated = pos, intensity, 0, real, False


class World:
    def __init__(self, mapdata: MapData, cfg: EnvConfig | None = None):
        self.map = mapdata
        self.cfg = cfg or EnvConfig()
        self.grid = mapdata.grid
        self.h, self.w = self.grid.shape
        self._dist_cache: dict[Pos, np.ndarray] = {}
        self._los_cache: dict[Pos, list[Pos]] = {}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        m = self.map
        self.t = 0
        self.hunter = m.hunter_start
        self.hunter_dir = (0, 1)
        self.hunter_scanning = False
        self.player = m.player_start
        self.player_hidden = False
        self.has_keycard = False
        self.terminal_restored = False
        self.alarm = False
        self.last_known: Pos | None = None
        self.last_known_age = 999
        self.last_known_investigated = True
        self.noises: list[Noise] = []
        self.trail = np.zeros((self.h, self.w), dtype=np.float32)
        self.visible = np.zeros((self.h, self.w), dtype=np.float32)
        self.player_visible = False
        self.checked_lockers: dict[Pos, int] = {}
        self.update_perception()

    # -------------------------------------------------------------- geometry
    def passable(self, p: Pos) -> bool:
        x, y = p
        return 0 <= x < self.w and 0 <= y < self.h and self.grid[y, x] != WALL

    def line_of_sight(self, a: Pos, b: Pos) -> bool:
        (x0, y0), (x1, y1) = a, b
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while (x0, y0) != (x1, y1):
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
            if (x0, y0) != (x1, y1) and self.grid[y0, x0] == WALL:
                return False
        return True

    def _los_tiles(self, origin: Pos) -> list[Pos]:
        """Tiles in line of sight within vision range (cached per tile)."""
        if origin not in self._los_cache:
            r = self.cfg.vision_range
            ox, oy = origin
            out = []
            for y in range(max(0, int(oy - r)), min(self.h, int(oy + r) + 1)):
                for x in range(max(0, int(ox - r)), min(self.w, int(ox + r) + 1)):
                    if (x - ox) ** 2 + (y - oy) ** 2 <= r * r and self.line_of_sight(origin, (x, y)):
                        out.append((x, y))
            self._los_cache[origin] = out
        return self._los_cache[origin]

    def in_view(self, origin: Pos, facing: tuple[int, int], target: Pos, scanning: bool) -> bool:
        dx, dy = target[0] - origin[0], target[1] - origin[1]
        d2 = dx * dx + dy * dy
        if d2 <= 2:
            return True
        if scanning:
            return d2 <= self.cfg.scan_range ** 2
        if d2 > self.cfg.vision_range ** 2:
            return False
        cos = (dx * facing[0] + dy * facing[1]) / math.sqrt(d2)
        return cos >= math.cos(math.radians(self.cfg.fov_deg / 2))

    def distance_field(self, target: Pos) -> np.ndarray:
        """BFS path distance from every tile to ``target`` (cached)."""
        if target not in self._dist_cache:
            dist = np.full((self.h, self.w), 10_000, dtype=np.int32)
            dist[target[1], target[0]] = 0
            q = deque([target])
            while q:
                x, y = q.popleft()
                for dx, dy in NEIGH:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h and self.grid[ny, nx] != WALL and dist[ny, nx] > dist[y, x] + 1:
                        dist[ny, nx] = dist[y, x] + 1
                        q.append((nx, ny))
            self._dist_cache[target] = dist
        return self._dist_cache[target]

    def path_dist(self, a: Pos, b: Pos) -> int:
        return int(self.distance_field(b)[a[1], a[0]])

    def step_toward(self, start: Pos, target: Pos, rng: np.random.Generator | None = None) -> Pos:
        dist = self.distance_field(target)
        best, cands = dist[start[1], start[0]], []
        for dx, dy in NEIGH:
            n = (start[0] + dx, start[1] + dy)
            if self.passable(n):
                d = dist[n[1], n[0]]
                if d < best:
                    best, cands = d, [n]
                elif d == best and cands:
                    cands.append(n)
        if not cands:
            return start
        return cands[int(rng.integers(len(cands)))] if (rng is not None and len(cands) > 1) else cands[0]

    # ------------------------------------------------------------- perception
    def update_perception(self):
        self.visible[:] = 0
        for (x, y) in self._los_tiles(self.hunter):
            if self.in_view(self.hunter, self.hunter_dir, (x, y), self.hunter_scanning):
                self.visible[y, x] = 1.0
        px, py = self.player
        seen = bool(self.visible[py, px])
        if seen and self.player_hidden:
            # a hidden player is only found by scanning right next to the locker
            seen = self.hunter_scanning and abs(px - self.hunter[0]) + abs(py - self.hunter[1]) <= 1
        self.player_visible = seen
        if seen:
            self.last_known, self.last_known_age = self.player, 0
            self.last_known_investigated = False
        else:
            self.last_known_age = min(self.last_known_age + 1, 999)

    def emit_noise(self, pos: Pos, intensity: float, real: bool = True):
        self.noises.append(Noise(pos, intensity, real))

    def heard_noises(self) -> list[Noise]:
        hx, hy = self.hunter
        out = []
        for n in self.noises:
            r = self.cfg.hearing_range * n.intensity
            if (n.pos[0] - hx) ** 2 + (n.pos[1] - hy) ** 2 <= r * r and not n.investigated:
                out.append(n)
        return out

    def loudest_heard(self) -> Noise | None:
        heard = self.heard_noises()
        if not heard:
            return None
        return max(heard, key=lambda n: n.intensity / (1 + 0.15 * n.age))

    def age_noises(self):
        for n in self.noises:
            n.age += 1
        self.noises = [n for n in self.noises if n.age < 25]

    # -------------------------------------------------------------- the hunter
    def resolve_target(self, action: int, behavior) -> Pos | None:
        if action == A_PURSUE:
            return self.last_known
        if action == A_INVESTIGATE:
            n = self.loudest_heard()
            return n.pos if n else None
        if action == A_INTERCEPT:
            exits = self.map.exits
            pred = behavior.predicted_exit(exits) if behavior is not None else None
            if pred is not None:
                return pred
            ref = self.last_known or self.hunter
            return min(exits, key=lambda e: self.path_dist(ref, e)) if exits else None
        if action == A_CHECK_HIDE:
            if not self.map.lockers:
                return None
            heat = behavior.hide_scores(self.map.lockers) if behavior is not None else {}

            def score(l):
                recently = 1.0 if self.t - self.checked_lockers.get(l, -99) < 30 else 0.0
                return heat.get(l, 0.0) - recently - 0.02 * self.path_dist(self.hunter, l)
            return max(self.map.lockers, key=score)
        return None

    def hunter_act(self, action: int, behavior=None, rng=None) -> dict:
        ev = {"bumped": False, "scanned": False, "no_target": False, "target": None}
        self.hunter_scanning = False
        if action in DIRS:
            d = DIRS[action]
            n = (self.hunter[0] + d[0], self.hunter[1] + d[1])
            self.hunter_dir = d
            if self.passable(n):
                self.hunter = n
            else:
                ev["bumped"] = True
        elif action == A_SCAN:
            self.hunter_scanning = True
            ev["scanned"] = True
        else:
            target = self.resolve_target(action, behavior)
            ev["target"] = target
            if target is None or target == self.hunter:
                self.hunter_scanning = True       # nothing to go to: look around
                ev["no_target"] = target is None
                ev["scanned"] = True
            else:
                n = self.step_toward(self.hunter, target, rng)
                if n != self.hunter:
                    self.hunter_dir = (n[0] - self.hunter[0], n[1] - self.hunter[1])
                    self.hunter = n
        hx, hy = self.hunter
        self.trail *= 0.97
        self.trail[hy, hx] = 1.0
        if self.grid[hy, hx] == LOCKER:
            self.checked_lockers[self.hunter] = self.t
        if self.hunter_scanning:
            for dx, dy in NEIGH:
                l = (hx + dx, hy + dy)
                if self.passable(l) and self.grid[l[1], l[0]] == LOCKER:
                    self.checked_lockers[l] = self.t
        return ev

    # -------------------------------------------------------------- the player
    def player_move(self, d: tuple[int, int]) -> bool:
        n = (self.player[0] + d[0], self.player[1] + d[1])
        if self.player_hidden or not self.passable(n):
            return False
        self.player = n
        if self.grid[n[1], n[0]] == KEYCARD and not self.has_keycard:
            self.has_keycard = True
        return True

    def toggle_hide(self) -> bool:
        x, y = self.player
        if self.grid[y, x] == LOCKER:
            self.player_hidden = not self.player_hidden
            return True
        return False

    def restore_terminal(self):
        if self.has_keycard and self.player == self.map.terminal and not self.terminal_restored:
            self.terminal_restored = True
            self.alarm = True                      # restoring power trips the alarm
            self.emit_noise(self.player, 1.6, real=True)
            return True
        return False

    def exit_unlocked(self) -> bool:
        return self.has_keycard and self.terminal_restored

    # ---------------------------------------------------------------- outcome
    def caught(self) -> bool:
        if self.hunter == self.player:
            return True
        adj = abs(self.hunter[0] - self.player[0]) + abs(self.hunter[1] - self.player[1]) == 1
        return adj and self.player_visible

    def escaped(self) -> bool:
        x, y = self.player
        return self.grid[y, x] == EXIT and self.exit_unlocked() and not self.player_hidden
