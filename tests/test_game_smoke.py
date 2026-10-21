"""Headless smoke test: the game boots, runs ticks and renders every scene."""
import pygame

from game.main import App
from game.scenes.screens import LabScene


def test_game_boots_and_renders():
    app = App("heuristic")
    scene = app.scene
    scene.story = False
    for _ in range(30):
        if app.scene is not scene:
            break
        scene.tick()
        scene.update(0.05)
        scene.draw(app.screen)
    app.goto_menu()
    app.scene.update(0.1)
    app.scene.draw(app.screen)
    LabScene(app).draw(app.screen)
    pygame.quit()
