"""Map generation and loading.

Procedural maps: a 3x2 grid of rooms joined by corridors (random spanning
tree plus extra loops), with lockers, 2-3 exits, a keycard and a terminal.
Hand-made maps: JSON files with ASCII rows (see ``assets/maps``).

Training uses seeds < 10_000; the held-out evaluation suite uses seeds
>= 10_000 so the hunter is always tested on layouts it never trained on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import EXIT, FLOOR, KEYCARD, LOCKER, MAP_H, MAP_W, TERMINAL, TILE_CHARS, WALL

EVAL_SEED_OFFSET = 10_000


@dataclass
class MapData:
    grid: np.ndarray                     # (H, W) int8 tile ids
    lockers: list[tuple[int, int]]
    exits: list[tuple[int, int]]
    keycard: tuple[int, int]
    terminal: tuple[int, int]
    player_start: tuple[int, int]
    hunter_start: tuple[int, int]
    rooms: list[tuple[int, int, int, int]] = field(default_factory=list)  # x, y, w, h
    name: str = "procedural"
    room_names: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def passable(self, x: int, y: int) -> bool:
        h, w = self.grid.shape
        return 0 <= x < w and 0 <= y < h and self.grid[y, x] != WALL

    def room_of(self, pos: tuple[int, int]) -> int | None:
        x, y = pos
        for i, (rx, ry, rw, rh) in enumerate(self.rooms):
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return i
        return None


ROOM_LABELS = ["Cryo Lab", "Server Hall", "Robotics Bay", "Archive", "Med Wing", "Control Room"]


def _carve(grid, x0, y0, x1, y1):
    grid[min(y0, y1):max(y0, y1) + 1, min(x0, x1):max(x0, x1) + 1] = FLOOR


def generate(seed: int, w: int = MAP_W, h: int = MAP_H) -> MapData:
    rng = np.random.default_rng(seed)
    grid = np.full((h, w), WALL, dtype=np.int8)
    cols, rows = 3, 2
    cw, ch = (w - 1) // cols, (h - 1) // rows  # 6 x 7 cells
    rooms = []
    for r in range(rows):
        for c in range(cols):
            rw = int(rng.integers(3, cw))          # 3..5
            rh = int(rng.integers(3, ch - 1))      # 3..5
            rx = 1 + c * cw + int(rng.integers(0, cw - rw))
            ry = 1 + r * ch + int(rng.integers(0, ch - rh))
            rooms.append((rx, ry, rw, rh))
            grid[ry:ry + rh, rx:rx + rw] = FLOOR

    def center(i):
        rx, ry, rw, rh = rooms[i]
        return rx + rw // 2, ry + rh // 2

    # adjacency of the 3x2 room grid
    edges = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                edges.append((i, i + 1))
            if r + 1 < rows:
                edges.append((i, i + cols))
    order = rng.permutation(len(edges))
    parent = list(range(len(rooms)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    chosen = []
    for k in order:
        a, b = edges[k]
        if find(a) != find(b):
            parent[find(a)] = find(b)
            chosen.append((a, b))
        elif rng.random() < 0.45:          # extra loops -> multiple escape routes
            chosen.append((a, b))
    for a, b in chosen:
        (ax, ay), (bx, by) = center(a), center(b)
        if rng.random() < 0.5:
            _carve(grid, ax, ay, bx, ay)
            _carve(grid, bx, ay, bx, by)
        else:
            _carve(grid, ax, ay, ax, by)
            _carve(grid, ax, by, bx, by)

    def room_tiles(i, wall_adjacent=False):
        rx, ry, rw, rh = rooms[i]
        out = []
        for y in range(ry, ry + rh):
            for x in range(rx, rx + rw):
                if grid[y, x] != FLOOR:
                    continue
                if wall_adjacent:
                    walls = sum(grid[y + dy, x + dx] == WALL for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                    if walls == 0 or walls >= 3:
                        continue
                out.append((x, y))
        return out

    start_room = int(rng.integers(len(rooms)))
    px, py = center(start_room)
    dists = [abs(center(i)[0] - px) + abs(center(i)[1] - py) for i in range(len(rooms))]
    hunter_room = int(np.argmax(dists))

    used: set[tuple[int, int]] = set()

    def pick(cands):
        cands = [c for c in cands if c not in used]
        p = cands[int(rng.integers(len(cands)))]
        used.add(p)
        return p

    player_start = pick(room_tiles(start_room))
    hunter_start = pick(room_tiles(hunter_room))

    lockers = []
    for i in range(len(rooms)):
        for _ in range(int(rng.integers(1, 3))):
            cands = [c for c in room_tiles(i, wall_adjacent=True) if c not in used]
            if cands:
                p = pick(cands)
                lockers.append(p)
                grid[p[1], p[0]] = LOCKER

    other = [i for i in range(len(rooms)) if i != start_room]
    exit_rooms = rng.choice([i for i in other if i != hunter_room], size=int(rng.integers(2, 4)), replace=False)
    exits = []
    for i in exit_rooms:
        p = pick(room_tiles(int(i)))
        exits.append(p)
        grid[p[1], p[0]] = EXIT
    key_room = int(rng.choice(other))
    keycard = pick(room_tiles(key_room))
    grid[keycard[1], keycard[0]] = KEYCARD
    term_room = int(rng.choice([i for i in range(len(rooms)) if i != key_room]))
    terminal = pick(room_tiles(term_room))
    grid[terminal[1], terminal[0]] = TERMINAL

    names = list(rng.permutation(ROOM_LABELS))
    return MapData(grid, lockers, exits, keycard, terminal, player_start, hunter_start,
                   rooms, name=f"proc-{seed}", room_names=names)


def load_json(path: str | Path) -> MapData:
    data = json.loads(Path(path).read_text())
    rows = data["rows"]
    h, w = len(rows), len(rows[0])
    grid = np.full((h, w), WALL, dtype=np.int8)
    lockers, exits = [], []
    keycard = terminal = player_start = hunter_start = None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            grid[y, x] = TILE_CHARS[ch]
            if ch == "L":
                lockers.append((x, y))
            elif ch == "E":
                exits.append((x, y))
            elif ch == "K":
                keycard = (x, y)
            elif ch == "T":
                terminal = (x, y)
            elif ch == "P":
                player_start = (x, y)
            elif ch == "H":
                hunter_start = (x, y)
    rooms = [tuple(r["rect"]) for r in data.get("rooms", [])]
    names = [r["name"] for r in data.get("rooms", [])]
    return MapData(grid, lockers, exits, keycard, terminal, player_start, hunter_start,
                   rooms, name=data.get("name", Path(path).stem), room_names=names)
