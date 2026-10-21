"""The hunter's brain inside the game.

Loads a saved PPO policy and runs inference locally (no training, no network
calls). The observation is built by exactly the same code used during
training, so the game hunter perceives the world the way it was trained to.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from ai.baselines import HabitHeuristic
from environment import observations
from environment.config import ACTION_NAMES

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
CKPT_DIR = ROOT / "ai" / "checkpoints"
DEFAULT_RUN = "ppo_bc_ft"


def find_checkpoints(run: str = DEFAULT_RUN) -> dict[str, Path]:
    d = CKPT_DIR / run
    out = {}
    for name in ("final", "best"):
        if (d / f"{name}.zip").exists():
            out["trained"] = d / f"{name}.zip"
            break
    ckpts = sorted(d.glob("ckpt_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    if ckpts:
        out["early"] = ckpts[0]
    return out


MODES = ["trained", "early", "heuristic"]
MODE_LABELS = {
    "trained": "Trained PPO hunter",
    "early": "Early checkpoint (barely trained)",
    "heuristic": "Scripted rules (no neural net)",
}


class HunterBrain:
    def __init__(self, mode: str = "trained", run: str = DEFAULT_RUN):
        self.mode = mode
        self.model = None
        self.label = MODE_LABELS.get(mode, mode)
        ckpts = find_checkpoints(run)
        if mode in ("trained", "early") and mode in ckpts:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(ckpts[mode]), device="cpu")
            self.label += f"  [{ckpts[mode].parent.name}/{ckpts[mode].stem}]"
        elif mode != "heuristic":
            self.mode = "heuristic"
            self.label = MODE_LABELS["heuristic"] + "  (no checkpoint found)"
        self.heuristic = HabitHeuristic(0)
        self.last_action = "scan"

    def act(self, world, behavior, step: int, max_steps: int = 300) -> int:
        if self.model is not None:
            obs = observations.build(world, behavior, min(step, max_steps - 1), max_steps)
            a = int(self.model.predict(obs, deterministic=True)[0])
        else:
            a = self.heuristic.predict(None, SimpleNamespace(world=world, behavior=behavior))
        self.last_action = ACTION_NAMES[a]
        return a

    def state_label(self, world) -> str:
        if world.player_visible:
            return "HUNTING"
        a = self.last_action
        if a == "pursue_last_known":
            return "TRACKING"
        if a == "investigate_noise":
            return "INVESTIGATING"
        if a == "intercept_exit":
            return "GUARDING EXIT"
        if a == "check_hiding_spot":
            return "CHECKING LOCKERS"
        if a == "scan":
            return "SCANNING"
        return "PATROLLING"
