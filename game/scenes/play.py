"""Chapter 1 gameplay scene."""
from __future__ import annotations

import pygame

from ai.player_behavior import BehaviorStore, PlayerBehaviorModel
from environment import maps
from environment.config import EnvConfig, LOCKER
from environment.world import World
from game.entities.hunter_brain import ROOT, HunterBrain
from game.systems.renderer import AMBER, DIM, GREEN, RED, TEAL, TILE, WHITE, Renderer, font

TICK = 0.2            # seconds per simulation step (hunter thinks once per tick)
HUD_H = 96
DATA_DIR = ROOT / "saves"
KEYS = {pygame.K_w: (0, -1), pygame.K_UP: (0, -1), pygame.K_s: (0, 1), pygame.K_DOWN: (0, 1),
        pygame.K_a: (-1, 0), pygame.K_LEFT: (-1, 0), pygame.K_d: (1, 0), pygame.K_RIGHT: (1, 0)}

STORY = [
    "CHAPTER 1  -  THE AWAKENING",
    "",
    "You regain consciousness inside a sealed laboratory.",
    "Something heavy is walking the corridors. Its eye glows red.",
    "",
    "Find the KEYCARD, restore the TERMINAL, reach an EXIT.",
    "It cannot see through walls - but it hears you, and it remembers.",
    "",
    "WASD move    SHIFT sprint (loud)    CTRL sneak (quiet)",
    "E hide / hold E at terminal    SPACE throw bottle    F1 hunter view",
    "",
    "Press ENTER to wake up",
]


class PlayScene:
    def __init__(self, app, hunter_mode: str = "trained", player: str = "player1"):
        self.app = app
        self.mapdata = maps.load_json(ROOT / "assets" / "maps" / "chapter1.json")
        self.cfg = EnvConfig()
        self.world = World(self.mapdata, self.cfg)
        self.store = BehaviorStore(DATA_DIR / "neurohunt.sqlite")
        self.player_name = player
        self.behavior: PlayerBehaviorModel = self.store.load(player, self.mapdata.name)
        self.brain = HunterBrain(hunter_mode)
        self.renderer = Renderer(self.world, (0, 0))
        self.story = True
        self.acc = 0.0
        self.t = 0.0
        self.step = 0
        self.stamina = 10.0
        self.bottles = 3
        self.terminal_progress = 0
        self.facing = (1, 0)
        self.message, self.message_t = "", 0.0
        self.prev_p = self.world.player
        self.prev_h = self.world.hunter
        self.sneak_toggle = 0
        self.f_big, self.f, self.f_small = font(30, True), font(16), font(13)
        self.heartbeat_t = 0.0
        self.alarm_played = False
        self.show_memory = False
        self.observer = False   # F1: see what the hunter sees (demo mode)

    # ------------------------------------------------------------- input
    def handle(self, ev):
        if ev.type != pygame.KEYDOWN:
            return
        if self.story:
            if ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.story = False
            elif ev.key == pygame.K_ESCAPE:
                self.app.goto_menu()
            return
        w = self.world
        if ev.key == pygame.K_ESCAPE:
            self.app.goto_menu()
        elif ev.key == pygame.K_e:
            if w.grid[w.player[1], w.player[0]] == LOCKER:
                w.toggle_hide()
                self.app.audio.play("locker")
                if w.player_hidden:
                    self.behavior.record_hide(w.player)
                    self.flash("Hidden. Hold your breath.")
            elif w.player == self.mapdata.terminal and not w.has_keycard:
                self.flash("Terminal locked - it needs a keycard.")
        elif ev.key == pygame.K_SPACE and not w.player_hidden:
            self.throw()
        elif ev.key == pygame.K_F1:
            self.observer = not self.observer
        elif ev.key == pygame.K_TAB:
            self.show_memory = not self.show_memory

    def flash(self, text):
        self.message, self.message_t = text, 2.5

    def throw(self):
        w = self.world
        if self.bottles <= 0:
            self.flash("No bottles left.")
            return
        x, y = w.player
        for _ in range(6):
            n = (x + self.facing[0], y + self.facing[1])
            if not w.passable(n):
                break
            x, y = n
        if (x, y) == w.player:
            return
        self.bottles -= 1
        w.emit_noise((x, y), 1.4, real=False)
        self.behavior.record_distraction()
        self.renderer.add_ripple((x, y), 1.4, AMBER)
        self.app.audio.play("throw")

    # ------------------------------------------------------------- simulation
    def tick(self):
        w = self.world
        keys = pygame.key.get_pressed()
        self.prev_p, self.prev_h = w.player, w.hunter
        w.t = self.step

        # --- hunter thinks and acts (local inference on the saved policy)
        action = self.brain.act(w, self.behavior, self.step)
        w.hunter_act(action, self.behavior)
        w.update_perception()
        if w.caught():
            return self.finish("caught")

        # --- player
        d = None
        for k, v in KEYS.items():
            if keys[k]:
                d = v
                break
        sprint = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        sneak = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] or keys[pygame.K_c]
        if d and not w.player_hidden:
            self.facing = d
            moves = 1
            noise = 0.55
            if sneak:
                self.sneak_toggle ^= 1
                moves = self.sneak_toggle
                noise = 0.2
            elif sprint and self.stamina >= 1:
                moves, noise = 2, 1.2
                self.stamina -= 1
            for _ in range(moves):
                if w.player_move(d):
                    self.behavior.record_move(d)
            if moves and w.player != self.prev_p:
                w.emit_noise(w.player, noise, real=True)
                if noise >= 0.5:
                    self.renderer.add_ripple(w.player, noise)
                    self.app.audio.play("step")
        if not (sprint and d):
            self.stamina = min(10.0, self.stamina + 0.15)
        if w.has_keycard and self.prev_p != w.player and w.player == self.mapdata.keycard:
            self.app.audio.play("pickup")
            self.flash("Keycard acquired. Find the terminal.")

        # terminal: hold E for 1.2 s
        if keys[pygame.K_e] and w.player == self.mapdata.terminal and w.has_keycard and not w.terminal_restored:
            self.terminal_progress += 1
            if self.terminal_progress >= 6:
                w.restore_terminal()
                self.app.audio.play("terminal")
                self.app.audio.play("alarm")
                self.renderer.add_ripple(w.player, 2.5, RED)
                self.flash("Power restored. ALARM TRIGGERED - run for an exit!")
        else:
            self.terminal_progress = 0

        w.update_perception()
        if w.caught():
            return self.finish("caught")
        if w.escaped():
            self.behavior.record_exit(w.player)
            return self.finish("escaped")
        w.age_noises()
        self.step += 1

    def finish(self, outcome):
        w = self.world
        if outcome == "caught" and w.alarm:
            # count the exit the player was clearly heading for as a (weaker) habit
            near = min(self.mapdata.exits, key=lambda e: w.path_dist(w.player, e))
            if w.path_dist(w.player, near) <= 6:
                self.behavior.record_exit(near, 0.5)
        self.behavior.end_episode()
        self.store.save(self.player_name, self.mapdata.name, self.behavior)
        self.store.log_run(self.player_name, self.mapdata.name, outcome, self.step, self.brain.mode)
        self.app.audio.play("caught" if outcome == "caught" else "escape")
        self.app.goto_end(outcome, self.step, self.behavior.describe(self.mapdata), self.brain.label,
                          self.store.stats(self.player_name))

    def update(self, dt):
        self.t += dt
        self.message_t = max(0.0, self.message_t - dt)
        if self.story:
            return
        self.acc += dt
        while self.acc >= TICK and self.app.scene is self:
            self.acc -= TICK
            self.tick()
        w = self.world
        dist = abs(w.hunter[0] - w.player[0]) + abs(w.hunter[1] - w.player[1])
        self.heartbeat_t -= dt
        if dist <= 6 and self.heartbeat_t <= 0:
            self.app.audio.play("heartbeat")
            self.heartbeat_t = 0.35 + dist * 0.12

    # ------------------------------------------------------------- drawing
    def _lerp(self, a, b):
        k = min(1.0, self.acc / TICK)
        return ((a[0] + (b[0] - a[0]) * k + 0.5) * TILE, (a[1] + (b[1] - a[1]) * k + 0.5) * TILE)

    def draw(self, surf):
        surf.fill((0, 0, 0))
        w = self.world
        if self.story:
            return self._draw_story(surf)
        heat = self.behavior.hide_heatmap(w.grid.shape, self.mapdata.lockers) if self.show_memory else None
        self.renderer.draw(surf, self._lerp(self.prev_p, w.player), self._lerp(self.prev_h, w.hunter),
                           w.hunter_dir, self.t, heat=heat, reveal_all=self.observer)
        if w.alarm and int(self.t * 2) % 2:
            red = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            red.fill((255, 0, 0, 18))
            surf.blit(red, (0, 0))
        self._draw_hud(surf)

    def _draw_story(self, surf):
        sw, sh = surf.get_size()
        for i, line in enumerate(STORY):
            f = self.f_big if i == 0 else self.f
            col = TEAL if i == 0 else (AMBER if line.startswith("Press") and int(self.t * 2) % 2 else WHITE)
            if "WASD" in line or line.startswith("E hide"):
                col = DIM
            img = f.render(line, True, col)
            surf.blit(img, (sw // 2 - img.get_width() // 2, 110 + i * 34))

    def _draw_hud(self, surf):
        w = self.world
        sw, sh = surf.get_size()
        y0 = sh - HUD_H
        pygame.draw.rect(surf, (8, 12, 15), (0, y0, sw, HUD_H))
        pygame.draw.line(surf, (40, 110, 120), (0, y0), (sw, y0), 2)
        objs = [("Keycard", w.has_keycard), ("Terminal", w.terminal_restored), ("Exit", False)]
        x = 16
        for name, done in objs:
            col = GREEN if done else DIM
            pygame.draw.rect(surf, col, (x, y0 + 16, 12, 12), 0 if done else 1)
            surf.blit(self.f.render(name, True, WHITE if not done else GREEN), (x + 20, y0 + 12))
            x += 130
        # stamina
        surf.blit(self.f_small.render("STAMINA", True, DIM), (16, y0 + 44))
        pygame.draw.rect(surf, (30, 40, 45), (90, y0 + 46, 150, 10))
        pygame.draw.rect(surf, TEAL, (90, y0 + 46, int(15 * self.stamina), 10))
        surf.blit(self.f_small.render(f"BOTTLES {self.bottles}", True, AMBER), (260, y0 + 43))
        surf.blit(self.f_small.render("TAB: what it remembers   F1: hunter view", True, DIM), (16, y0 + 70))
        # hunter panel
        state = self.brain.state_label(w)
        col = RED if state in ("HUNTING", "TRACKING") else (AMBER if state != "PATROLLING" else DIM)
        surf.blit(self.f_small.render("HUNTER", True, DIM), (430, y0 + 12))
        surf.blit(self.f.render(state, True, col), (500, y0 + 9))
        surf.blit(self.f_small.render(self.brain.label[:58], True, DIM), (430, y0 + 40))
        habits = self.behavior.describe(self.mapdata)[0]
        surf.blit(self.f_small.render(habits[:60], True, (200, 120, 120)), (430, y0 + 64))
        if self.message_t > 0:
            img = self.f.render(self.message, True, AMBER)
            bg = pygame.Rect(0, 0, img.get_width() + 24, img.get_height() + 12)
            bg.center = (sw // 2, 28)
            pygame.draw.rect(surf, (10, 14, 18), bg, border_radius=6)
            pygame.draw.rect(surf, (80, 70, 30), bg, 1, border_radius=6)
            surf.blit(img, (bg.x + 12, bg.y + 6))
