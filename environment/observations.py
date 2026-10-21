"""Hunter observations.

``grid`` is a multi-channel top-down image of what the hunter knows (fed to a
CNN); ``vec`` holds compact scalar features. The hunter never sees the
player's true position unless the player is in its current field of view.

Channels:
 0 walls                        5 behaviour prior: where this player hides
 1 hunter position              6 exits (+ predicted exit weight)
 2 currently visible tiles      7 hunter's own recent trail
 3 player (only if perceived)   8 lockers checked recently
 4 last known player position (fades with age) + heard noises
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .config import WALL
from .world import World

N_CHANNELS = 9
VEC_DIM = 16


def observation_space(h: int, w: int) -> spaces.Dict:
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(N_CHANNELS, h, w), dtype=np.float32),
        "vec": spaces.Box(-1.0, 1.0, shape=(VEC_DIM,), dtype=np.float32),
    })


def build(world: World, behavior, step: int, max_steps: int) -> dict:
    h, w = world.h, world.w
    g = np.zeros((N_CHANNELS, h, w), dtype=np.float32)
    g[0] = world.grid == WALL
    hx, hy = world.hunter
    g[1, hy, hx] = 1.0
    g[2] = world.visible
    if world.player_visible:
        g[3, world.player[1], world.player[0]] = 1.0
    if world.last_known is not None:
        lx, ly = world.last_known
        g[4, ly, lx] = max(g[4, ly, lx], 1.0 / (1.0 + world.last_known_age / 10.0))
    for n in world.heard_noises():
        x, y = n.pos
        g[4, y, x] = max(g[4, y, x], min(1.0, n.intensity) * 0.8 / (1.0 + n.age / 5.0))
    if behavior is not None:
        g[5] = behavior.hide_heatmap((h, w), world.map.lockers)
        pred, conf = behavior.exit_confidence(world.map.exits)
    else:
        pred, conf = None, 0.0
    for e in world.map.exits:
        g[6, e[1], e[0]] = 0.5 + (0.5 * conf if e == pred else 0.0)
    g[7] = world.trail
    for (lx, ly), t in world.checked_lockers.items():
        g[8, ly, lx] = max(0.0, 1.0 - (world.t - t) / 30.0)

    loud = world.loudest_heard()
    v = np.zeros(VEC_DIM, dtype=np.float32)
    v[0], v[1] = hx / (w - 1), hy / (h - 1)
    dx, dy = world.hunter_dir
    v[2 + {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}[(dx, dy)]] = 1.0
    v[6] = float(world.player_visible)
    v[7] = min(world.last_known_age, 100) / 100.0
    v[8] = float(world.alarm)
    v[9] = float(loud is not None)
    v[10] = step / max_steps
    if behavior is not None:
        v[11:16] = behavior.summary_vector(world.map.lockers, world.map.exits)
    return {"grid": g, "vec": v}
