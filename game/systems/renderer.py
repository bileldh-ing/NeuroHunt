"""Top-down renderer: dark sci-fi lab, teal lights, the hunter's red vision cone,
fog of war from the player's point of view, and noise ripples."""
from __future__ import annotations

import math
import random

import pygame

from environment.config import EXIT, KEYCARD, LOCKER, TERMINAL, WALL

TILE = 40
BG = (6, 9, 12)
FLOOR_A = (20, 27, 31)
FLOOR_B = (24, 32, 37)
WALL_C = (38, 46, 54)
WALL_EDGE = (70, 190, 200)
TEAL = (60, 220, 230)
RED = (255, 50, 60)
AMBER = (255, 190, 60)
GREEN = (80, 255, 140)
WHITE = (225, 235, 240)
DIM = (120, 140, 150)


def font(size: int, bold: bool = False) -> pygame.font.Font:
    for name in ("dejavusansmono", "consolas", "menlo", "couriernew"):
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size + 6)


_GLOWS: dict = {}


def glow(radius: int, color, strength: float = 1.0) -> pygame.Surface:
    """Radial light falloff, meant to be blitted with BLEND_RGB_ADD."""
    key = (radius, tuple(color), strength)
    if key not in _GLOWS:
        s = pygame.Surface((radius * 2, radius * 2))
        s.fill((0, 0, 0))
        steps = max(8, radius // 3)
        for i in range(steps, 0, -1):
            r = radius * i / steps
            k = strength * (1 - i / steps) ** 2
            pygame.draw.circle(s, tuple(int(c * k) for c in color), (radius, radius), int(r))
        _GLOWS[key] = s
    return _GLOWS[key]


class Renderer:
    def __init__(self, world, origin=(0, 0)):
        self.world = world
        self.ox, self.oy = origin
        self.w_px, self.h_px = world.w * TILE, world.h * TILE
        self.static = self._build_static()
        self.seen = [[False] * world.w for _ in range(world.h)]
        self._pvis_cache: dict = {}
        self.ripples: list[list] = []   # x, y, radius, max_radius, color
        self.flicker = 0.0
        rnd = random.Random(3)
        self.dust = [(rnd.uniform(0, self.w_px), rnd.uniform(0, self.h_px), rnd.uniform(0.2, 0.8)) for _ in range(70)]

    # ------------------------------------------------------------ static map
    def _build_static(self) -> pygame.Surface:
        w = self.world
        s = pygame.Surface((self.w_px, self.h_px))
        s.fill(BG)
        rnd = random.Random(7)
        for y in range(w.h):
            for x in range(w.w):
                r = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
                if w.grid[y, x] == WALL:
                    pygame.draw.rect(s, WALL_C, r)
                    shade = rnd.randint(-4, 4)
                    pygame.draw.rect(s, (30 + shade, 37 + shade, 44 + shade), r.inflate(-6, -6))
                    for dx, dy, edge in ((0, 1, "b"), (0, -1, "t"), (1, 0, "r"), (-1, 0, "l")):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w.w and 0 <= ny < w.h and w.grid[ny, nx] != WALL:
                            if edge == "b":
                                pygame.draw.line(s, WALL_EDGE, r.bottomleft, r.bottomright, 2)
                            elif edge == "t":
                                pygame.draw.line(s, (40, 110, 120), r.topleft, r.topright, 1)
                            elif edge == "r":
                                pygame.draw.line(s, (40, 110, 120), r.topright, r.bottomright, 1)
                            else:
                                pygame.draw.line(s, (40, 110, 120), r.topleft, r.bottomleft, 1)
                else:
                    pygame.draw.rect(s, FLOOR_A if (x + y) % 2 else FLOOR_B, r)
                    pygame.draw.rect(s, (14, 19, 22), r, 1)
                    if rnd.random() < 0.12:
                        gx, gy = r.x + rnd.randint(6, 26), r.y + rnd.randint(6, 26)
                        pygame.draw.rect(s, (16, 22, 26), (gx, gy, 10, 10))
                        for i in range(3):
                            pygame.draw.line(s, (30, 40, 45), (gx + 1, gy + 2 + i * 3), (gx + 9, gy + 2 + i * 3))
        # ceiling light pools
        for (rx, ry, rw, rh) in w.map.rooms:
            cx, cy = (rx + rw / 2) * TILE, (ry + rh / 2) * TILE
            g = glow(TILE * 3, (40, 150, 160), 0.45)
            s.blit(g, (cx - TILE * 3, cy - TILE * 3), special_flags=pygame.BLEND_RGB_ADD)
        return s

    # ------------------------------------------------------------ visibility
    def player_visible_tiles(self) -> set:
        w = self.world
        key = w.player
        if key not in self._pvis_cache:
            px, py = key
            r = 7
            vis = set()
            for y in range(max(0, py - r), min(w.h, py + r + 1)):
                for x in range(max(0, px - r), min(w.w, px + r + 1)):
                    if (x - px) ** 2 + (y - py) ** 2 <= r * r and w.line_of_sight(key, (x, y)):
                        vis.add((x, y))
            self._pvis_cache[key] = vis
        return self._pvis_cache[key]

    def add_ripple(self, tile, intensity, color=TEAL):
        cx, cy = (tile[0] + 0.5) * TILE, (tile[1] + 0.5) * TILE
        self.ripples.append([cx, cy, 4.0, max(20.0, intensity * 6 * TILE * 0.6), color])

    # ------------------------------------------------------------------ draw
    def tile_center(self, pos):
        return self.ox + (pos[0] + 0.5) * TILE, self.oy + (pos[1] + 0.5) * TILE

    def draw(self, surf, player_px, hunter_px, hunter_dir, t, show_hunter_cone=True, reveal_all=False, heat=None):
        w = self.world
        surf.blit(self.static, (self.ox, self.oy))
        vis = self.player_visible_tiles()
        for (x, y) in vis:
            self.seen[y][x] = True

        # objects
        for y in range(w.h):
            for x in range(w.w):
                tile = w.grid[y, x]
                if tile in (LOCKER, EXIT, KEYCARD, TERMINAL):
                    self._draw_object(surf, tile, x, y, t)

        # behaviour heat (debug / "what it remembers" view)
        if heat is not None:
            for y in range(w.h):
                for x in range(w.w):
                    if heat[y, x] > 0.01:
                        a = int(160 * min(1.0, heat[y, x]))
                        hs = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
                        hs.fill((255, 60, 60, a))
                        surf.blit(hs, (self.ox + x * TILE, self.oy + y * TILE))

        hunter_seen = reveal_all or w.hunter in vis
        # hunter vision cone
        if show_hunter_cone and hunter_seen:
            cone = pygame.Surface((self.w_px, self.h_px), pygame.SRCALPHA)
            col = (255, 40, 50, 55) if not w.hunter_scanning else (255, 170, 40, 60)
            for y in range(w.h):
                for x in range(w.w):
                    if w.visible[y, x] and ((x, y) in vis or reveal_all):
                        cone.fill(col, (x * TILE, y * TILE, TILE, TILE))
            surf.blit(cone, (self.ox, self.oy))

        # player
        self._draw_player(surf, player_px, t, w.player_hidden)
        if hunter_seen:
            self._draw_hunter(surf, hunter_px, hunter_dir, t)

        # ripples
        for rp in self.ripples:
            rp[2] += 2.4
            alpha = max(0, int(200 * (1 - rp[2] / rp[3])))
            if alpha:
                rs = pygame.Surface((rp[3] * 2 + 4, rp[3] * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(rs, (*rp[4], alpha), (rp[3] + 2, rp[3] + 2), int(rp[2]), 2)
                surf.blit(rs, (self.ox + rp[0] - rp[3] - 2, self.oy + rp[1] - rp[3] - 2))
        self.ripples = [r for r in self.ripples if r[2] < r[3]]

        # floating dust for atmosphere
        for i, (dx, dy, sp) in enumerate(self.dust):
            yy = (dy + t * 8 * sp) % self.h_px
            xx = (dx + math.sin(t * sp + i) * 6) % self.w_px
            surf.fill((70, 110, 120), (self.ox + xx, self.oy + yy, 2, 2))

        # fog of war
        if not reveal_all:
            fog = pygame.Surface((self.w_px, self.h_px), pygame.SRCALPHA)
            px, py = w.player
            for y in range(w.h):
                for x in range(w.w):
                    if (x, y) in vis:
                        d = math.hypot(x - px, y - py)
                        a = int(min(150, max(0, (d - 3) * 30)))
                    elif self.seen[y][x]:
                        a = 205
                    else:
                        a = 250
                    if a:
                        fog.fill((2, 4, 6, a), (x * TILE, y * TILE, TILE, TILE))
            surf.blit(fog, (self.ox, self.oy))

    def _draw_object(self, surf, tile, x, y, t):
        w = self.world
        r = pygame.Rect(self.ox + x * TILE, self.oy + y * TILE, TILE, TILE)
        if tile == LOCKER:
            body = r.inflate(-10, -4)
            pygame.draw.rect(surf, (52, 64, 72), body, border_radius=3)
            pygame.draw.rect(surf, (90, 120, 130), body, 1, border_radius=3)
            for i in range(3):
                pygame.draw.line(surf, (30, 38, 44), (body.x + 4, body.y + 5 + i * 4), (body.right - 4, body.y + 5 + i * 4))
            occupied = w.player_hidden and w.player == (x, y)
            pygame.draw.circle(surf, TEAL if occupied else (40, 80, 90), (body.right - 5, body.centery + 6), 2)
        elif tile == EXIT:
            open_ = w.exit_unlocked()
            col = GREEN if open_ else RED
            pygame.draw.rect(surf, (25, 30, 32), r.inflate(-4, -4), border_radius=4)
            for i in range(4):
                yy = r.y + 8 + i * 7
                pygame.draw.line(surf, col if (i + int(t * 3)) % 2 else (60, 70, 70), (r.x + 7, yy), (r.right - 7, yy), 3)
            pygame.draw.rect(surf, col, r.inflate(-4, -4), 2, border_radius=4)
        elif tile == KEYCARD and not w.has_keycard:
            pulse = 0.5 + 0.5 * math.sin(t * 4)
            card = pygame.Rect(0, 0, 20, 14)
            card.center = r.center
            surf.blit(glow(26, (255, 190, 60), 0.4 + 0.4 * pulse), (r.centerx - 26, r.centery - 26), special_flags=pygame.BLEND_RGB_ADD)
            pygame.draw.rect(surf, AMBER, card, border_radius=2)
            pygame.draw.rect(surf, (90, 60, 10), (card.x + 3, card.y + 3, 6, 5))
        elif tile == TERMINAL:
            on = w.terminal_restored
            scr = r.inflate(-8, -12)
            pygame.draw.rect(surf, (20, 24, 26), r.inflate(-4, -4), border_radius=3)
            pygame.draw.rect(surf, (40, 255, 160) if on else (90, 30, 30), scr)
            if not on and int(t * 2) % 2:
                pygame.draw.rect(surf, (200, 60, 60), scr, 1)
            for i in range(3):
                pygame.draw.line(surf, (10, 40, 30) if on else (40, 10, 10), (scr.x + 2, scr.y + 3 + i * 4), (scr.x + 2 + (i * 7 + 8) % scr.w, scr.y + 3 + i * 4))

    def _draw_player(self, surf, pos, t, hidden):
        x, y = pos
        if hidden:
            return
        surf.blit(glow(70, (40, 200, 220), 0.55), (x - 70, y - 70), special_flags=pygame.BLEND_RGB_ADD)
        pygame.draw.circle(surf, (15, 20, 24), (x, y), 13)
        pygame.draw.circle(surf, (35, 45, 55), (x, y), 11)
        pygame.draw.circle(surf, TEAL, (x, y), 11, 2)
        pygame.draw.circle(surf, (190, 250, 255), (x, y - 3), 4)  # visor

    def _draw_hunter(self, surf, pos, d, t):
        x, y = pos
        ang = math.atan2(d[1], d[0])
        surf.blit(glow(80, (255, 30, 40), 0.7), (x - 80, y - 80), special_flags=pygame.BLEND_RGB_ADD)
        pts = []
        for k in range(6):
            a = ang + k * math.pi / 3
            rr = 17 if k % 2 == 0 else 14
            pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr))
        pygame.draw.polygon(surf, (24, 26, 30), pts)
        pygame.draw.polygon(surf, (110, 30, 35), pts, 2)
        ex, ey = x + math.cos(ang) * 9, y + math.sin(ang) * 9
        pulse = 0.6 + 0.4 * math.sin(t * 10)
        pygame.draw.circle(surf, (int(255 * pulse), 40, 40), (ex, ey), 4)
        pygame.draw.circle(surf, (255, 200, 200), (ex, ey), 1)
        for side in (-1, 1):
            a = ang + side * 1.9
            pygame.draw.line(surf, (60, 64, 70), (x + math.cos(a) * 12, y + math.sin(a) * 12),
                             (x + math.cos(a) * 21, y + math.sin(a) * 21), 3)
