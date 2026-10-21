"""Title menu, end screen and the AI Lab screen (learning curves + metrics)."""
from __future__ import annotations

import math

import pygame

from game.entities.hunter_brain import MODE_LABELS, MODES, ROOT, find_checkpoints
from game.systems.renderer import AMBER, DIM, GREEN, RED, TEAL, WHITE, font, glow


class MenuScene:
    def __init__(self, app):
        self.app = app
        self.items = ["Play Chapter 1", "Hunter", "AI Lab", "Forget my habits", "Quit"]
        self.sel = 0
        self.mode_i = 0
        self.t = 0.0
        self.f_title, self.f, self.f_small = font(64, True), font(20), font(14)
        self.note = ""
        ck = find_checkpoints()
        if "trained" not in ck:
            self.mode_i = MODES.index("heuristic")

    def handle(self, ev):
        if ev.type != pygame.KEYDOWN:
            return
        if ev.key in (pygame.K_UP, pygame.K_w):
            self.sel = (self.sel - 1) % len(self.items)
        elif ev.key in (pygame.K_DOWN, pygame.K_s):
            self.sel = (self.sel + 1) % len(self.items)
        elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_a, pygame.K_d) and self.items[self.sel] == "Hunter":
            self.mode_i = (self.mode_i + (1 if ev.key in (pygame.K_RIGHT, pygame.K_d) else -1)) % len(MODES)
        elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
            item = self.items[self.sel]
            if item == "Play Chapter 1":
                self.app.goto_play(MODES[self.mode_i])
            elif item == "Hunter":
                self.mode_i = (self.mode_i + 1) % len(MODES)
            elif item == "AI Lab":
                self.app.scene = LabScene(self.app)
            elif item == "Forget my habits":
                from ai.player_behavior import BehaviorStore
                from game.scenes.play import DATA_DIR
                BehaviorStore(DATA_DIR / "neurohunt.sqlite").reset("player1")
                self.note = "The hunter's memory of you has been wiped."
            elif item == "Quit":
                self.app.running = False
        elif ev.key == pygame.K_ESCAPE:
            self.app.running = False

    def update(self, dt):
        self.t += dt

    def draw(self, surf):
        sw, sh = surf.get_size()
        surf.fill((4, 6, 9))
        for i in range(0, sw, 40):
            pygame.draw.line(surf, (10, 16, 20), (i, 0), (i, sh))
        for j in range(0, sh, 40):
            pygame.draw.line(surf, (10, 16, 20), (0, j), (sw, j))
        # scanning red eye
        ex = sw // 2 + int(math.sin(self.t * 0.8) * 220)
        surf.blit(glow(160, (255, 30, 40), 0.55), (ex - 160, 40), special_flags=pygame.BLEND_RGB_ADD)
        pygame.draw.circle(surf, (255, 90, 90), (ex, 200), 5)
        title = self.f_title.render("NEUROHUNT", True, TEAL)
        surf.blit(title, (sw // 2 - title.get_width() // 2, 150))
        sub = self.f.render("The Enemy That Learns You", True, (200, 90, 95))
        surf.blit(sub, (sw // 2 - sub.get_width() // 2, 230))
        for i, item in enumerate(self.items):
            label = item
            if item == "Hunter":
                label = f"Hunter:  < {MODE_LABELS[MODES[self.mode_i]]} >"
            col = AMBER if i == self.sel else WHITE
            img = self.f.render(("> " if i == self.sel else "  ") + label, True, col)
            surf.blit(img, (sw // 2 - 230, 320 + i * 44))
        if self.note:
            img = self.f_small.render(self.note, True, GREEN)
            surf.blit(img, (sw // 2 - img.get_width() // 2, 560))
        foot = self.f_small.render("PPO hunter trained with Stable-Baselines3 - inference runs locally, offline", True, DIM)
        surf.blit(foot, (sw // 2 - foot.get_width() // 2, sh - 40))


class EndScene:
    def __init__(self, app, outcome, steps, notes, hunter_label, stats):
        self.app, self.outcome, self.steps, self.notes, self.hunter_label, self.stats = app, outcome, steps, notes, hunter_label, stats
        self.f_big, self.f, self.f_small = font(48, True), font(18), font(14)
        self.t = 0.0

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_RETURN, pygame.K_r):
                self.app.goto_play(self.app.last_mode)
            elif ev.key == pygame.K_ESCAPE:
                self.app.goto_menu()

    def update(self, dt):
        self.t += dt

    def draw(self, surf):
        sw, sh = surf.get_size()
        surf.fill((6, 4, 6) if self.outcome == "caught" else (3, 8, 7))
        title = "IT FOUND YOU" if self.outcome == "caught" else "YOU ESCAPED SECTOR A"
        img = self.f_big.render(title, True, RED if self.outcome == "caught" else GREEN)
        surf.blit(img, (sw // 2 - img.get_width() // 2, 140))
        lines = [f"Survived {self.steps * 0.2:.1f} s  ({self.steps} ticks)",
                 f"Opponent: {self.hunter_label}",
                 f"Record: {self.stats.get('escaped', 0)} escapes / {self.stats.get('caught', 0)} captures",
                 "", "WHAT THE HUNTER NOW KNOWS ABOUT YOU:"] + ["  - " + n for n in self.notes]
        for i, line in enumerate(lines):
            col = AMBER if line.startswith("WHAT") else (WHITE if not line.startswith("  -") else (230, 140, 140))
            surf.blit(self.f.render(line, True, col), (120, 240 + i * 32))
        hint = self.f_small.render("ENTER play again     ESC menu", True, DIM)
        surf.blit(hint, (sw // 2 - hint.get_width() // 2, sh - 60))


class LabScene:
    """Shows the evidence that the model really was trained."""

    def __init__(self, app):
        self.app = app
        self.f, self.f_small = font(18, True), font(13)
        self.imgs = []
        for name in ("learning_curve.png", "baseline_comparison.png"):
            p = ROOT / "experiments" / "learning_curves" / name
            if p.exists():
                self.imgs.append(pygame.image.load(str(p)))
        md = ROOT / "experiments" / "metrics" / "ppo_bc_ft" / "comparison.md"
        self.table = md.read_text().split("\n\n")[0].splitlines() if md.exists() else ["Run `python -m ai.evaluate --compare` to generate metrics."]

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            self.app.goto_menu()

    def update(self, dt):
        pass

    def draw(self, surf):
        sw, sh = surf.get_size()
        surf.fill((5, 8, 10))
        surf.blit(self.f.render("AI LAB  -  how the hunter learned", True, TEAL), (24, 14))
        y = 44
        for img in self.imgs:
            scale = (sw - 40) / img.get_width()
            im = pygame.transform.smoothscale(img, (sw - 40, int(img.get_height() * scale)))
            surf.blit(im, (20, y))
            y += im.get_height() + 6
        note = "Full metrics table: experiments/metrics/ppo_bc_ft/comparison.md"
        surf.blit(self.f_small.render(note, True, DIM), (20, sh - 44))
        surf.blit(self.f_small.render("any key: back", True, DIM), (20, sh - 24))
