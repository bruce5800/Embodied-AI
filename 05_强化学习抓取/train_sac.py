"""SAC 训练脚本。

用法：
    # M1: 单物体 baseline（默认 red_cylinder, 500k step）
    python train_sac.py

    # 指定其他物体
    python train_sac.py --target blue_cube

    # M2: 多物体随机
    python train_sac.py --target random --total-timesteps 1_000_000 --run-name m2_random

    # 短跑验证 pipeline
    python train_sac.py --total-timesteps 5_000 --run-name smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from env import GraspEnv


REPO_ROOT = Path(__file__).resolve().parent


def make_env(target_object, max_episode_steps, seed):
    """构造单个 GraspEnv（包 Monitor 用于 EvalCallback）。"""
    env = GraspEnv(
        target_object=target_object,
        max_episode_steps=max_episode_steps,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def parse_target(target_arg: str):
    """'random' → None；其他直接当 body name。"""
    return None if target_arg == "random" else target_arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sac_default.yaml")
    parser.add_argument("--target", default="red_cylinder",
                        help="固定目标物体；'random' 表示每 episode 随机")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="覆盖 config 里的 total_timesteps")
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="m1_red",
                        help="日志/检查点子目录名")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--ckptdir", default="checkpoints")
    args = parser.parse_args()

    # ── 读 config ──
    with open(REPO_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    target = parse_target(args.target)
    total_steps = args.total_timesteps or int(cfg["total_timesteps"])

    log_dir = REPO_ROOT / args.logdir / args.run_name
    ckpt_dir = REPO_ROOT / args.ckptdir / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("─" * 60)
    print(f"  run_name        : {args.run_name}")
    print(f"  target          : {args.target}  (None=random per episode)")
    print(f"  total_timesteps : {total_steps:,}")
    print(f"  max_ep_steps    : {args.max_episode_steps}")
    print(f"  log_dir         : {log_dir}")
    print(f"  ckpt_dir        : {ckpt_dir}")
    print("─" * 60)

    # ── env ──
    train_env = make_env(target, args.max_episode_steps, seed=args.seed)
    eval_env = make_env(target, args.max_episode_steps, seed=args.seed + 10_000)

    # ── 模型 ──
    model = SAC(
        cfg["policy"],
        train_env,
        learning_rate=float(cfg["learning_rate"]),
        buffer_size=int(cfg["buffer_size"]),
        batch_size=int(cfg["batch_size"]),
        tau=float(cfg["tau"]),
        gamma=float(cfg["gamma"]),
        ent_coef=cfg["ent_coef"],
        target_entropy=cfg["target_entropy"],
        learning_starts=int(cfg["learning_starts"]),
        gradient_steps=int(cfg["gradient_steps"]),
        train_freq=int(cfg["train_freq"]),
        policy_kwargs=cfg.get("policy_kwargs", {}),
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
    )

    # ── 回调 ──
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir),
        log_path=str(log_dir),
        eval_freq=int(cfg["eval_freq"]),
        n_eval_episodes=int(cfg["n_eval_episodes"]),
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=int(cfg["checkpoint_freq"]),
        save_path=str(ckpt_dir),
        name_prefix="sac",
    )

    # ── 训练 ──
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([eval_cb, ckpt_cb]),
        progress_bar=True,
    )

    final_path = ckpt_dir / "final.zip"
    model.save(final_path)
    print(f"\nfinal model saved → {final_path}")

    # 简短的最终评估
    print("\n最终评估（10 episode, deterministic）:")
    rewards, lengths = [], []
    n_succ, n_lift = 0, 0
    for ep in range(10):
        obs, info = eval_env.reset(seed=99_000 + ep)
        ep_r, ep_lift = 0.0, False
        for t in range(args.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(action)
            ep_r += r
            ep_lift = ep_lift or info.get("lifted", False)
            if term or trunc:
                break
        rewards.append(ep_r)
        lengths.append(t + 1)
        n_succ += int(info.get("placed", False))
        n_lift += int(ep_lift)
    print(f"  success_rate : {n_succ/10:.0%}")
    print(f"  lift_rate    : {n_lift/10:.0%}")
    print(f"  mean_reward  : {np.mean(rewards):+.2f}")
    print(f"  mean_length  : {np.mean(lengths):.1f}")


if __name__ == "__main__":
    main()
