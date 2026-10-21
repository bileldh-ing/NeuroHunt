"""Player behaviour model: the part of NeuroHunt that "learns you".

The PPO hunter is a fixed, versioned network. Individual adaptation comes from
this separate component, which tracks a player's habits across games:

* where they hide (per locker),
* which exit they escape through,
* how often they throw distractions,
* how often they change direction.

It turns those habits into a compact summary (a heat-map channel, a predicted
exit and a 5-number vector) that is part of the hunter's observation. The
network was trained to *use* that summary, so a player who hides in the same
locker four times will find the hunter checking that locker first - without
the network's weights ever changing during play.

Habits are persisted in SQLite between sessions (``BehaviorStore``).
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

Pos = tuple[int, int]


def _key(p: Pos) -> str:
    return f"{p[0]},{p[1]}"


def _unkey(s: str) -> Pos:
    x, y = s.split(",")
    return int(x), int(y)


class PlayerBehaviorModel:
    def __init__(self, decay: float = 0.85):
        self.decay = decay
        self.hide_counts: dict[Pos, float] = {}
        self.exit_counts: dict[Pos, float] = {}
        self.distractions = 0.0
        self.moves = 0.0
        self.dir_changes = 0.0
        self.episodes = 0
        self._last_dir: tuple[int, int] | None = None

    # ------------------------------------------------------------ recording
    def record_hide(self, pos: Pos, weight: float = 1.0):
        self.hide_counts[pos] = self.hide_counts.get(pos, 0.0) + weight

    def record_exit(self, pos: Pos, weight: float = 1.0):
        self.exit_counts[pos] = self.exit_counts.get(pos, 0.0) + weight

    def record_distraction(self):
        self.distractions += 1

    def record_move(self, d: tuple[int, int]):
        self.moves += 1
        if self._last_dir is not None and d != self._last_dir:
            self.dir_changes += 1
        self._last_dir = d

    def end_episode(self):
        """Older habits fade so the hunter tracks the player's *current* style."""
        self.episodes += 1
        self.hide_counts = {k: v * self.decay for k, v in self.hide_counts.items() if v * self.decay > 0.05}
        self.exit_counts = {k: v * self.decay for k, v in self.exit_counts.items() if v * self.decay > 0.05}
        self.distractions *= self.decay
        self.moves *= self.decay
        self.dir_changes *= self.decay
        self._last_dir = None

    # ------------------------------------------------------------ querying
    def hide_scores(self, lockers: list[Pos]) -> dict[Pos, float]:
        total = sum(self.hide_counts.get(l, 0.0) for l in lockers)
        if total <= 0:
            return {}
        return {l: self.hide_counts.get(l, 0.0) / total for l in lockers}

    def hide_heatmap(self, shape: tuple[int, int], lockers: list[Pos]) -> np.ndarray:
        heat = np.zeros(shape, dtype=np.float32)
        scores = self.hide_scores(lockers)
        if scores:
            conf = 1 - math.exp(-sum(self.hide_counts.get(l, 0.0) for l in lockers) / 2.0)
            for (x, y), s in scores.items():
                heat[y, x] = s * conf
        return heat

    def exit_confidence(self, exits: list[Pos]) -> tuple[Pos | None, float]:
        total = sum(self.exit_counts.get(e, 0.0) for e in exits)
        if total <= 0:
            return None, 0.0
        best = max(exits, key=lambda e: self.exit_counts.get(e, 0.0))
        share = self.exit_counts.get(best, 0.0) / total
        conf = share * (1 - math.exp(-total / 2.0))
        return best, conf

    def predicted_exit(self, exits: list[Pos], min_conf: float = 0.25) -> Pos | None:
        best, conf = self.exit_confidence(exits)
        return best if conf >= min_conf else None

    def summary_vector(self, lockers: list[Pos], exits: list[Pos]) -> np.ndarray:
        scores = self.hide_scores(lockers)
        top_hide = max(scores.values()) if scores else 0.0
        _, exit_conf = self.exit_confidence(exits)
        distract_rate = self.distractions / max(1.0, self.moves / 50.0)  # per 50 moves
        turn_rate = self.dir_changes / max(1.0, self.moves)
        return np.array([top_hide, exit_conf, min(distract_rate, 1.0), turn_rate,
                         min(self.episodes, 20) / 20.0], dtype=np.float32)

    def describe(self, mapdata) -> list[str]:
        """Human-readable notes shown on the game-over screen."""
        notes = []
        scores = self.hide_scores(mapdata.lockers)
        if scores:
            l, s = max(scores.items(), key=lambda kv: kv[1])
            room = mapdata.room_of(l)
            name = mapdata.room_names[room] if room is not None and room < len(mapdata.room_names) else "a locker"
            if s > 0.4:
                notes.append(f"It checks the locker in {name} first ({s:.0%} of your hides).")
        best, conf = self.exit_confidence(mapdata.exits)
        if best is not None and conf > 0.25:
            room = mapdata.room_of(best)
            name = mapdata.room_names[room] if room is not None and room < len(mapdata.room_names) else "one exit"
            notes.append(f"It expects you to run for the exit in {name}.")
        if self.distractions > 2:
            notes.append("It has started to doubt noises you make on purpose.")
        return notes or ["It hasn't found a pattern in your play yet."]

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "decay": self.decay,
            "hide_counts": {_key(k): v for k, v in self.hide_counts.items()},
            "exit_counts": {_key(k): v for k, v in self.exit_counts.items()},
            "distractions": self.distractions, "moves": self.moves,
            "dir_changes": self.dir_changes, "episodes": self.episodes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerBehaviorModel":
        m = cls(d.get("decay", 0.85))
        m.hide_counts = {_unkey(k): v for k, v in d.get("hide_counts", {}).items()}
        m.exit_counts = {_unkey(k): v for k, v in d.get("exit_counts", {}).items()}
        m.distractions = d.get("distractions", 0.0)
        m.moves = d.get("moves", 0.0)
        m.dir_changes = d.get("dir_changes", 0.0)
        m.episodes = d.get("episodes", 0)
        return m


class BehaviorStore:
    """SQLite persistence: one behaviour profile per (player, map)."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS behavior (player TEXT, map TEXT, data TEXT, updated REAL,"
            " PRIMARY KEY (player, map))")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, player TEXT, map TEXT,"
            " outcome TEXT, steps INTEGER, hunter TEXT, ts REAL)")
        self.conn.commit()

    def load(self, player: str, map_name: str) -> PlayerBehaviorModel:
        row = self.conn.execute("SELECT data FROM behavior WHERE player=? AND map=?", (player, map_name)).fetchone()
        return PlayerBehaviorModel.from_dict(json.loads(row[0])) if row else PlayerBehaviorModel()

    def save(self, player: str, map_name: str, model: PlayerBehaviorModel):
        self.conn.execute("INSERT OR REPLACE INTO behavior VALUES (?,?,?,?)",
                          (player, map_name, json.dumps(model.to_dict()), time.time()))
        self.conn.commit()

    def log_run(self, player: str, map_name: str, outcome: str, steps: int, hunter: str):
        self.conn.execute("INSERT INTO runs (player, map, outcome, steps, hunter, ts) VALUES (?,?,?,?,?,?)",
                          (player, map_name, outcome, steps, hunter, time.time()))
        self.conn.commit()

    def reset(self, player: str):
        self.conn.execute("DELETE FROM behavior WHERE player=?", (player,))
        self.conn.commit()

    def stats(self, player: str) -> dict:
        rows = self.conn.execute("SELECT outcome, COUNT(*) FROM runs WHERE player=? GROUP BY outcome", (player,)).fetchall()
        return dict(rows)
