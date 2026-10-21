"""Plot learning curves and the baseline comparison.

    python -m ai.plot_curves --run ppo_cnn_v1
Writes experiments/learning_curves/learning_curve.png and baseline_comparison.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG, FG, TEAL, RED, AMBER = "#06090c", "#cfe3e8", "#3cdce6", "#ff3c46", "#ffbe3c"


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#2a3a40")
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(color="#1a262b", lw=0.6)


def learning_curve(run: str, out: Path):
    mdir = ROOT / "experiments" / "metrics" / run
    rows = list(csv.DictReader(open(mdir / "train_episodes.csv")))
    t = np.array([int(r["timesteps"]) for r in rows])
    win = np.array([r["outcome"] == "caught" for r in rows], dtype=float)
    esc = np.array([r["outcome"] == "escaped" for r in rows], dtype=float)
    k = max(50, len(rows) // 60)
    smooth = lambda x: np.convolve(x, np.ones(k) / k, mode="valid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG)
    ax = axes[0]
    _style(ax)
    ax.plot(t[k - 1:], smooth(win), color=RED, lw=2, label="hunter catches player (train)")
    ax.plot(t[k - 1:], smooth(esc), color=TEAL, lw=1.5, label="player escapes (train)")
    ev_path = mdir / "eval_history.csv"
    if ev_path.exists():
        ev = list(csv.DictReader(open(ev_path)))
        if ev:
            ax.plot([int(r["timesteps"]) for r in ev], [float(r["hunter_win_rate"]) for r in ev], "o--",
                    color=AMBER, lw=1.5, label="win rate, held-out maps")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_title("PPO hunter - learning curve")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor="#2a3a40", fontsize=8)
    ax = axes[1]
    _style(ax)
    caught = [(int(r["timesteps"]), int(r["steps"])) for r in rows if r["outcome"] == "caught"]
    if caught:
        ct, cs = np.array(caught).T
        kk = max(20, len(cs) // 60)
        ax.plot(ct[kk - 1:], np.convolve(cs, np.ones(kk) / kk, mode="valid"), color=RED, lw=2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("steps until capture")
    ax.set_title("Time to capture (lower = more efficient hunter)")
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=110, facecolor=BG)
    plt.close(fig)


def comparison(run: str, out: Path):
    p = ROOT / "experiments" / "metrics" / run / "comparison.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    names = list(data)
    wr = [data[n]["aggregate"]["hunter_win_rate"]["mean"] for n in names]
    sd = [data[n]["aggregate"]["hunter_win_rate"]["std"] for n in names]
    ttc = [data[n]["aggregate"]["mean_time_to_capture"]["mean"] or 0 for n in names]
    colors = [RED if n.startswith("ppo_final") or n.startswith("ppo_best") else (AMBER if n.startswith("ppo") else "#4a6a72") for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG)
    for ax, vals, title, err in ((axes[0], wr, "Hunter win rate on unseen maps (3 seeds)", sd),
                                 (axes[1], ttc, "Mean steps to capture", None)):
        _style(ax)
        ax.bar(range(len(names)), vals, color=colors, yerr=err, ecolor=FG, capsize=4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split(" (")[0] for n in names], fontsize=8, color=FG, rotation=30, ha="right")
        ax.set_title(title)
    axes[0].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "baseline_comparison.png", dpi=110, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_cnn_v1")
    args = ap.parse_args()
    out = ROOT / "experiments" / "learning_curves"
    out.mkdir(parents=True, exist_ok=True)
    learning_curve(args.run, out)
    comparison(args.run, out)
    print("plots written to", out)


if __name__ == "__main__":
    main()
