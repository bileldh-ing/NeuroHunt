"""Behaviour-cloning warm start for the PPO hunter.

Pure PPO from scratch (run ppo_cnn_v1) collapsed onto a single intent within
50k steps on this sparse-reward task. A standard fix is to initialise the
policy by imitating a competent scripted hunter, then let PPO fine-tune it on
the real reward. This script:

  1. plays whole episodes with the HabitHeuristic hunter on training maps,
  2. trains the SB3 policy's actor (cross-entropy on the expert action) and
     critic (Monte-Carlo discounted return), streaming in chunks so memory
     stays small,
  3. saves a regular PPO .zip that `ai.train --init` fine-tunes.

    python -m ai.pretrain_bc --config experiments/configs/ppo_bc_ft.json --samples 60000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ai.baselines import HabitHeuristic
from ai.networks import NeuroHuntExtractor
from environment.config import EnvConfig
from environment.neurohunt_env import NeuroHuntEnv, make_env

ROOT = Path(__file__).resolve().parents[1]


def collect(env, expert, n_min, gamma):
    grids, vecs, acts, rets = [], [], [], []
    while len(acts) < n_min:
        obs, _ = env.reset()
        ep_r, done = [], False
        while not done:
            a = expert.predict(obs, env)
            grids.append(obs["grid"].astype(np.float16))
            vecs.append(obs["vec"])
            acts.append(a)
            obs, r, te, tr, _ = env.step(a)
            ep_r.append(r)
            done = te or tr
        g, out = 0.0, []
        for r in reversed(ep_r):
            g = r + gamma * g
            out.append(g)
        rets.extend(reversed(out))
    return np.stack(grids).astype(np.float32), np.stack(vecs), np.array(acts), np.array(rets, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/ppo_bc_ft.json"))
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(1)
    conf = json.loads(Path(args.config).read_text())
    cfg = EnvConfig.from_json(args.config)
    run = conf["run_name"]
    run_dir = ROOT / "ai" / "checkpoints" / run
    mdir = ROOT / "experiments" / "metrics" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    pol = conf["policy"]
    venv = DummyVecEnv([make_env(cfg, rank=0, seed=conf["seed"] * 1000)])
    model = PPO("MultiInputPolicy", venv, seed=conf["seed"], device="cpu",
                policy_kwargs=dict(features_extractor_class=NeuroHuntExtractor,
                                   features_extractor_kwargs=dict(cnn_dim=pol["cnn_dim"], vec_dim=pol["vec_dim"]),
                                   net_arch=dict(pi=pol["net_arch"], vf=pol["net_arch"])),
                **conf["ppo"])
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    env = NeuroHuntEnv(cfg)
    env.reset(seed=conf["seed"] + 77)
    expert = HabitHeuristic(conf["seed"])
    log, seen = [], 0
    while seen < args.samples:
        g, v, a, ret = collect(env, expert, args.chunk, conf["ppo"]["gamma"])
        seen += len(a)
        n = len(a)
        for _ in range(args.epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, 256):
                b = idx[s:s + 256]
                obs = {"grid": torch.as_tensor(g[b]), "vec": torch.as_tensor(v[b])}
                values, logp, ent = policy.evaluate_actions(obs, torch.as_tensor(a[b]))
                loss_pi = -logp.mean()
                loss_v = torch.nn.functional.mse_loss(values.flatten(), torch.as_tensor(ret[b]))
                loss = loss_pi + 0.5 * loss_v - 0.001 * ent.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
        with torch.no_grad():
            pred = policy.get_distribution({"grid": torch.as_tensor(g), "vec": torch.as_tensor(v)}).distribution.probs.argmax(1).numpy()
        acc = float((pred == a).mean())
        log.append({"samples": seen, "nll": float(loss_pi), "value_mse": float(loss_v), "train_acc": acc})
        print(f"[bc] samples={seen} nll={float(loss_pi):.3f} vmse={float(loss_v):.4f} acc={acc:.3f}", flush=True)
        if len(log) % 5 == 0:
            model.save(run_dir / "bc_init")
    model.save(run_dir / "bc_init")
    (mdir / "bc_log.json").write_text(json.dumps(log, indent=2))
    print("saved", run_dir / "bc_init.zip")


if __name__ == "__main__":
    main()
