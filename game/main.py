"""NeuroHunt - entry point.

    python -m game.main            # play
    python -m game.main --hunter early      # face the barely-trained checkpoint
    python -m game.main --hunter heuristic  # face hand-written rules
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):           # PyInstaller bundle
    sys.path.insert(0, getattr(sys, "_MEIPASS"))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame  # noqa: E402

from game.scenes.play import HUD_H, PlayScene  # noqa: E402
from game.scenes.screens import EndScene, MenuScene  # noqa: E402
from game.systems.audio import Audio  # noqa: E402
from game.systems.renderer import TILE  # noqa: E402

WIDTH, HEIGHT = 21 * TILE, 15 * TILE + HUD_H


class App:
    def __init__(self, hunter_mode: str | None = None):
        pygame.init()
        pygame.display.set_caption("NeuroHunt - The Enemy That Learns You")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.audio = Audio()
        self.running = True
        self.last_mode = hunter_mode or "trained"
        self.scene = PlayScene(self, hunter_mode) if hunter_mode else MenuScene(self)

    def goto_menu(self):
        self.scene = MenuScene(self)

    def goto_play(self, mode):
        self.last_mode = mode
        self.scene = PlayScene(self, mode)

    def goto_end(self, outcome, steps, notes, label, stats):
        self.scene = EndScene(self, outcome, steps, notes, label, stats)

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                else:
                    self.scene.handle(ev)
            self.scene.update(dt)
            self.scene.draw(self.screen)
            pygame.display.flip()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hunter", choices=["trained", "early", "heuristic"])
    args = ap.parse_args()
    App(args.hunter).run()


if __name__ == "__main__":
    main()
