"""Held-out evaluation suite.

Every hunter (PPO checkpoints and scripted baselines) plays the same
episodes: unseen procedural maps (seeds >= 10_000), all player profiles
including the held-out "ghost", across several random seeds.

Metrics: hunter win rate, mean time to capture, escape rate, timeout rate,
interception rate, false investigations per episode.

Usage:
    python -m ai.evaluate --model ai/checkpoints/ppo_cnn_v1/final.zip --episodes 100 --seeds 0 1 2
    python -m ai.evaluate --compare   # all baselines + early/final checkpoints -> experiments/metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai.baselines import BASELINES
from environment.config import EnvConfig
from environment.neurohunt_env import NeuroHuntEnv

ROOT = Path(__file__).resolve().parents[1]


class PPOHunter:
    def __init__(self, path: str | Path, deterministic: bool = True):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(path), device="cpu")
        self.deterministic = deterministic
        self.name = Path(path).stem

    def predict(self, obs, env=None):
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a)


def run_episodes(policy, episodes: int, seed: int, cfg: EnvConfig | None = None, map_path: str | None = None) -> dict:
    env = NeuroHuntEnv(cfg, eval_mode=True, map_path=map_path)
    env.reset(seed=seed)
    outcomes, capture_t, false_inv, intercepts, per_profile = [], [], [], [], {}
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            obs, _, te, tr, info = env.step(policy.predict(obs, env))
            done = te or tr
        outcomes.append(info["outcome"])
        if info["outcome"] == "caught":
            capture_t.append(info["steps"])
        false_inv.append(info["false_investigations"])
        intercepts.append(info["reward_breakdown"]["intercept"] > 0)
        per_profile.setdefault(info["profile"], []).append(info["outcome"] == "caught")
    n = len(outcomes)
    return {
        "episodes": n,
        "hunter_win_rate": outcomes.count("caught") / n,
        "escape_rate": outcomes.count("escaped") / n,
        "timeout_rate": outcomes.count("timeout") / n,
        "mean_time_to_capture": float(np.mean(capture_t)) if capture_t else None,
        "interception_rate": float(np.mean(intercepts)),
        "false_investigations_per_ep": float(np.mean(false_inv)),
        "win_rate_by_profile": {k: float(np.mean(v)) for k, v in sorted(per_profile.items())},
    }


def evaluate(policy, episodes: int = 100, seeds=(0, 1, 2), cfg=None, map_path=None) -> dict:
    per_seed = [run_episodes(policy, episodes, s, cfg, map_path) for s in seeds]
    keys = ["hunter_win_rate", "escape_rate", "timeout_rate", "interception_rate", "false_investigations_per_ep"]
    agg = {k: {"mean": float(np.mean([r[k] for r in per_seed])), "std": float(np.std([r[k] for r in per_seed]))} for k in keys}
    ttc = [r["mean_time_to_capture"] for r in per_seed if r["mean_time_to_capture"] is not None]
    agg["mean_time_to_capture"] = {"mean": float(np.mean(ttc)) if ttc else None, "std": float(np.std(ttc)) if ttc else None}
    profiles = per_seed[0]["win_rate_by_profile"].keys()
    agg["win_rate_by_profile"] = {p: float(np.mean([r["win_rate_by_profile"].get(p, 0) for r in per_seed])) for p in profiles}
    return {"aggregate": agg, "per_seed": per_seed, "seeds": list(seeds), "episodes_per_seed": episodes}


def compare(run_dir: Path, episodes: int, seeds, out_dir: Path, cfg=None, also=()):
    results = {}
    for name, cls in BASELINES.items():
        print(f"evaluating baseline {name} ...", flush=True)
        results[name] = evaluate(cls(0), episodes, seeds, cfg)
    ckpts = sorted(run_dir.glob("ckpt_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    picks = {}
    for item in also:                      # label=path pairs, e.g. other runs for ablations
        label, path = item.split("=", 1)
        picks[label] = Path(path)
    if (run_dir / "bc_init.zip").exists():
        picks["imitation_only (bc_init)"] = run_dir / "bc_init.zip"
    if ckpts:
        picks[f"ppo_early ({ckpts[0].stem})"] = ckpts[0]
    if (run_dir / "final.zip").exists():
        picks["ppo_final"] = run_dir / "final.zip"
    best = run_dir / "best.zip"
    if best.exists():
        picks["ppo_best"] = best
    for label, path in picks.items():
        print(f"evaluating {label} ...", flush=True)
        results[label] = evaluate(PPOHunter(path), episodes, seeds, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(results, indent=2))
    lines = ["| Hunter | Win rate | Escape rate | Timeout | Time to capture | Interceptions | False investigations/ep |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        a = r["aggregate"]
        ttc = a["mean_time_to_capture"]["mean"]
        lines.append(f"| {name} | {a['hunter_win_rate']['mean']:.1%} ± {a['hunter_win_rate']['std']:.1%} | "
                     f"{a['escape_rate']['mean']:.1%} | {a['timeout_rate']['mean']:.1%} | "
                     f"{ttc:.1f} steps | {a['interception_rate']['mean']:.1%} | {a['false_investigations_per_ep']['mean']:.2f} |"
                     if ttc else f"| {name} | {a['hunter_win_rate']['mean']:.1%} | - | - | - | - | - |")
    prof_names = list(next(iter(results.values()))["aggregate"]["win_rate_by_profile"].keys())
    lines += ["", "| Hunter | " + " | ".join(prof_names) + " |", "|---|" + "---|" * len(prof_names)]
    for name, r in results.items():
        wr = r["aggregate"]["win_rate_by_profile"]
        lines.append(f"| {name} | " + " | ".join(f"{wr[p]:.0%}" for p in prof_names) + " |")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--baseline", choices=list(BASELINES))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--run", default="ppo_cnn_v1")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--also", nargs="*", default=[], help="extra label=path.zip models to include in --compare")
    ap.add_argument("--map", help="evaluate on a fixed JSON map instead of procedural ones")
    args = ap.parse_args()
    if args.compare:
        compare(ROOT / "ai" / "checkpoints" / args.run, args.episodes, args.seeds, ROOT / "experiments" / "metrics" / args.run, also=args.also)
        return
    policy = PPOHunter(args.model) if args.model else BASELINES[args.baseline or "patrol_chaser"](0)
    print(json.dumps(evaluate(policy, args.episodes, args.seeds, map_path=args.map)["aggregate"], indent=2))


if __name__ == "__main__":
    main()
