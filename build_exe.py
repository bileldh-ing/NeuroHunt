"""Package NeuroHunt into a standalone executable with PyInstaller.

    pip install pyinstaller
    python build_exe.py

The bundle contains the game, the map files and the trained checkpoints, so
the hunter runs fully offline on the player's machine.
"""
import os
import PyInstaller.__main__

sep = os.pathsep
PyInstaller.__main__.run([
    "game/main.py",
    "--name=NeuroHunt",
    "--onedir",
    "--windowed",
    f"--add-data=assets{sep}assets",
    f"--add-data=ai/checkpoints{sep}ai/checkpoints",
    f"--add-data=experiments/learning_curves{sep}experiments/learning_curves",
    f"--add-data=experiments/metrics{sep}experiments/metrics",
    "--hidden-import=stable_baselines3",
    "--collect-submodules=environment",
    "--collect-submodules=ai",
    "--collect-submodules=game",
    "--noconfirm",
])
