"""Train the NeuroHunt hunter with PPO (Stable-Baselines3).

    python -m ai.train --config experiments/configs/ppo_cnn_v1.json
    python -m ai.train --config experiments/configs/ppo_cnn_v1.json --resume ai/checkpoints/ppo_cnn_v1/final.zip

Outputs (per run):
  ai/checkpoints/<run>/ckpt_<steps>_steps.zip   versioned weights
  ai/checkpoints/<run>/best.zip / final.zip
  ai/checkpoints/<run>/config.json              exact config used
  experiments/metrics/<run>/train_episodes.csv  every finished training episode
  experiments/metrics/<run>/eval_history.csv    held-out evaluation over time
  experiments/metrics/<run>/tb/                 TensorBoard logs
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ai.evaluate import run_episodes
from ai.networks import NeuroHuntExtractor
from environment.config import EnvConfig
from environment.neurohunt_env import make_env

ROOT = Path(__file__).resolve().parents[1]


class _PolicyAdapter:
    def __init__(self, model):
        self.model = model

    def predict(self, obs, env=None):
        return int(self.model.predict(obs, deterministic=True)[0])


class NeuroHuntCallback(BaseCallback):
    def __init__(self, run_dir: Path, metrics_dir: Path, cfg: EnvConfig, ckpt_every: int, eval_every: int, eval_episodes: int):
        super().__init__()
        self.run_dir, self.metrics_dir, self.cfg = run_dir, metrics_dir, cfg
        self.ckpt_every, self.eval_every, self.eval_episodes = ckpt_every, eval_every, eval_episodes
        self.next_ckpt, self.next_eval = ckpt_every, eval_every
        self.best = -1.0
        self.t0 = time.time()
        self.recent: list[dict] = []
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self.ep_file = open(metrics_dir / "train_episodes.csv", "a", newline="")
        self.ep_writer = csv.writer(self.ep_file)
        if self.ep_file.tell() == 0:
            self.ep_writer.writerow(["timesteps", "outcome", "steps", "profile", "return_unscaled", "intercept", "false_investigations"])
        self.ev_file = open(metrics_dir / "eval_history.csv", "a", newline="")
        self.ev_writer = csv.writer(self.ev_file)
        if self.ev_file.tell() == 0:
            self.ev_writer.writerow(["timesteps", "hunter_win_rate", "escape_rate", "timeout_rate", "mean_time_to_capture",
                                     "interception_rate", "false_investigations_per_ep", "wall_clock_s"])

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "outcome" in info:
                rb = info["reward_breakdown"]
                self.ep_writer.writerow([self.num_timesteps, info["outcome"], info["steps"], info["profile"],
                                         round(sum(rb.values()), 2), int(rb["intercept"] > 0), info["false_investigations"]])
                self.recent.append(info)
        if len(self.recent) >= 200:
            wins = np.mean([i["outcome"] == "caught" for i in self.recent])
            self.logger.record("neurohunt/train_win_rate", wins)
            self.logger.record("neurohunt/train_ep_len", np.mean([i["steps"] for i in self.recent]))
            self.recent = []
        if self.num_timesteps >= self.next_ckpt:
            self.model.save(self.run_dir / f"ckpt_{self.num_timesteps}_steps")
            self.next_ckpt += self.ckpt_every
            self.ep_file.flush()
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            r = run_episodes(_PolicyAdapter(self.model), self.eval_episodes, seed=123, cfg=self.cfg)
            self.ev_writer.writerow([self.num_timesteps, r["hunter_win_rate"], r["escape_rate"], r["timeout_rate"],
                                     r["mean_time_to_capture"], r["interception_rate"], r["false_investigations_per_ep"],
                                     round(time.time() - self.t0)])
            self.ev_file.flush()
            self.logger.record("neurohunt/eval_win_rate", r["hunter_win_rate"])
            print(f"[eval] t={self.num_timesteps} win={r['hunter_win_rate']:.2f} escape={r['escape_rate']:.2f} "
                  f"ttc={r['mean_time_to_capture']}", flush=True)
            if r["hunter_win_rate"] > self.best:
                self.best = r["hunter_win_rate"]
                self.model.save(self.run_dir / "best")
        return True

    def _on_training_end(self):
        self.ep_file.close()
        self.ev_file.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/ppo_cnn_v1.json"))
    ap.add_argument("--timesteps", type=int, help="override total_timesteps")
    ap.add_argument("--resume", help="continue from a saved .zip, or 'auto' for the latest checkpoint")
    ap.add_argument("--init", help="start from pretrained weights (e.g. the behaviour-cloning bc_init.zip)")
    ap.add_argument("--subproc", action="store_true", help="use SubprocVecEnv")
    args = ap.parse_args()

    conf = json.loads(Path(args.config).read_text())
    cfg = EnvConfig.from_json(args.config)
    run = conf["run_name"]
    run_dir = ROOT / "ai" / "checkpoints" / run
    metrics_dir = ROOT / "experiments" / "metrics" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.json")
    torch.set_num_threads(conf.get("torch_threads", 1))

    seed = conf.get("seed", 0)
    fns = [make_env(cfg, rank=i, seed=seed * 1000) for i in range(conf["n_envs"])]
    venv = SubprocVecEnv(fns) if args.subproc else DummyVecEnv(fns)
    pol = conf["policy"]
    policy_kwargs = dict(features_extractor_class=NeuroHuntExtractor,
                         features_extractor_kwargs=dict(cnn_dim=pol["cnn_dim"], vec_dim=pol["vec_dim"]),
                         net_arch=dict(pi=pol["net_arch"], vf=pol["net_arch"]))
    total = args.timesteps or conf["total_timesteps"]
    if args.resume == "auto":
        cands = list(run_dir.glob("ckpt_*_steps.zip"))
        args.resume = str(max(cands, key=lambda p: int(p.stem.split("_")[1]))) if cands else None
        if args.resume:
            print("resuming from", args.resume, flush=True)
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=str(metrics_dir / "tb"))
        reset = False
        total = max(0, total - model.num_timesteps)   # remaining budget
    elif args.init:
        lr = conf["ppo"]["learning_rate"]
        model = PPO.load(args.init, env=venv, device="cpu", tensorboard_log=str(metrics_dir / "tb"),
                         custom_objects={**conf["ppo"], "lr_schedule": (lambda _: lr)})
        model.set_random_seed(seed)
        reset = True
    else:
        model = PPO("MultiInputPolicy", venv, policy_kwargs=policy_kwargs, seed=seed, device="cpu", verbose=0,
                    tensorboard_log=str(metrics_dir / "tb"), **conf["ppo"])
        reset = True
    cb = NeuroHuntCallback(run_dir, metrics_dir, cfg, conf["checkpoint_every"], conf["eval_every"], conf["eval_episodes"])
    if not reset:
        cb.next_ckpt = (model.num_timesteps // cb.ckpt_every + 1) * cb.ckpt_every
        cb.next_eval = (model.num_timesteps // cb.eval_every + 1) * cb.eval_every
        if (metrics_dir / "eval_history.csv").exists():
            rows = list(csv.DictReader(open(metrics_dir / "eval_history.csv")))
            cb.best = max((float(r["hunter_win_rate"]) for r in rows), default=-1.0)
    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)   # time-boxed runs still save their weights
    try:
        model.learn(total_timesteps=total, callback=cb, reset_num_timesteps=reset, tb_log_name=run)
    except KeyboardInterrupt:
        print("interrupted - saving", flush=True)
        model.save(run_dir / f"ckpt_{model.num_timesteps}_steps")
    model.save(run_dir / "final")
    print("saved", run_dir / "final.zip")


if __name__ == "__main__":
    main()
