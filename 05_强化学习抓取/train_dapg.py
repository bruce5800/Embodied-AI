"""train_dapg.py — SAC + BC regularization (DAPG-style) 训练脚本。

关键设计：
  - 用 SACWithBC（SAC actor loss 加 BC reg）替代 vanilla SAC
  - demos 同时灌进:
      a) replay_buffer (跟 SAC 收集的 rollout 混合，给 critic 学)
      b) demo_buffer  (固定的 demo 池，给 actor BC reg 用)
  - λ schedule: bc_lambda_init → bc_lambda_min over training
  - 跟 train_sac.py 用同样的 8-worker VecEnv + EvalCallback

用法：
    python train_dapg.py --demos demos/blue_cube_v10_strict.npz \\
        --target blue_cube --total-timesteps 200000 --n-envs 8 \\
        --run-name m1_dapg_blue
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import GraspEnv
from sac_with_bc import SACWithBC


REPO_ROOT = Path(__file__).resolve().parent


def make_env_fn(target, max_steps, seed, soften):
    def _init():
        env = GraspEnv(
            target_object=target,
            max_episode_steps=max_steps,
            soften_contacts=soften,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sac_default.yaml")
    parser.add_argument("--demos", required=True,
                        help="npz 文件（collect_demos.py 输出）")
    parser.add_argument("--target", default="blue_cube")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="m1_dapg")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--ckptdir", default="checkpoints")

    parser.add_argument("--obj-radius", type=float, default=None)
    parser.add_argument("--soften-contacts", action="store_true",
                        help="env 启用 soften（如果 demo 是软接触下收集的，必须开）")

    # DAPG 超参
    parser.add_argument("--bc-lambda-init", type=float, default=1.0)
    parser.add_argument("--bc-lambda-decay", type=float, default=0.99995,
                        help="每 gradient step 衰减；200k step 衰减到约 0.0067 起始 → 配合 min")
    parser.add_argument("--bc-lambda-min", type=float, default=0.05)
    parser.add_argument("--bc-close-weight", type=float, default=3.0,
                        help="close action 样本权重（防 mode collapse）")

    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    # config
    with open(REPO_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    target = None if args.target == "random" else args.target
    log_dir = REPO_ROOT / args.logdir / args.run_name
    ckpt_dir = REPO_ROOT / args.ckptdir / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("─" * 60)
    print(f"  run_name        : {args.run_name}")
    print(f"  target          : {args.target}")
    print(f"  demos           : {args.demos}")
    print(f"  total_timesteps : {args.total_timesteps:,}")
    print(f"  n_envs          : {args.n_envs}")
    print(f"  obj_radius      : {args.obj_radius}")
    print(f"  soften_contacts : {args.soften_contacts}")
    print(f"  bc_lambda_init  : {args.bc_lambda_init}")
    print(f"  bc_lambda_decay : {args.bc_lambda_decay}")
    print(f"  bc_lambda_min   : {args.bc_lambda_min}")
    print(f"  bc_close_weight : {args.bc_close_weight}")
    print("─" * 60)

    # ── env (vec) ──
    fns_train = [
        make_env_fn(target, args.max_episode_steps, args.seed + i,
                    args.soften_contacts)
        for i in range(args.n_envs)
    ]
    if args.n_envs > 1:
        train_env = SubprocVecEnv(fns_train, start_method="spawn")
    else:
        train_env = DummyVecEnv(fns_train)

    # eval env (单进程)
    eval_env_fn = make_env_fn(target, args.max_episode_steps,
                              args.seed + 10_000, args.soften_contacts)
    eval_env = eval_env_fn()

    # ── LR schedule ──
    base_lr = args.lr if args.lr is not None else float(cfg["learning_rate"])
    def lr_schedule(progress_remaining: float) -> float:
        return base_lr * max(progress_remaining, 0.1)

    grad_steps = int(cfg["gradient_steps"])

    # ── 创建 SACWithBC ──
    model = SACWithBC(
        cfg["policy"],
        train_env,
        learning_rate=lr_schedule,
        buffer_size=int(cfg["buffer_size"]),
        batch_size=int(cfg["batch_size"]),
        tau=float(cfg["tau"]),
        gamma=float(cfg["gamma"]),
        ent_coef=cfg["ent_coef"],
        target_entropy=cfg["target_entropy"],
        learning_starts=int(cfg["learning_starts"]),
        gradient_steps=grad_steps,
        train_freq=int(cfg["train_freq"]),
        policy_kwargs=cfg.get("policy_kwargs", {}),
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
        # SAC + BC 特有
        bc_lambda_init=args.bc_lambda_init,
        bc_lambda_decay=args.bc_lambda_decay,
        bc_lambda_min=args.bc_lambda_min,
        bc_close_weight=args.bc_close_weight,
    )

    # ── 加载 demos: 灌进 demo_buffer (BC reg 用) + replay_buffer (critic 用) ──
    print(f"  → loading demos {args.demos}")
    model.load_demos_from_npz(args.demos)

    # 同时灌进 replay_buffer (按 vec batch)
    data = np.load(args.demos)
    n_demo = len(data["obs"])
    n_envs = model.replay_buffer.n_envs
    n_full = (n_demo // n_envs) * n_envs
    for start in range(0, n_full, n_envs):
        end = start + n_envs
        model.replay_buffer.add(
            obs=data["obs"][start:end],
            next_obs=data["next_obs"][start:end],
            action=data["action"][start:end],
            reward=data["reward"][start:end].astype(np.float32),
            done=data["done"][start:end].astype(np.float32),
            infos=[{} for _ in range(n_envs)],
        )
    print(f"  → replay_buffer also loaded {n_full:,} demo transitions")
    model.learning_starts = 0   # buffer 有数据，不要 random rollout

    # ── 回调 ──
    eval_freq_vec = max(int(cfg["eval_freq"]) // max(args.n_envs, 1), 1)
    ckpt_freq_vec = max(int(cfg["checkpoint_freq"]) // max(args.n_envs, 1), 1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq_vec,
        n_eval_episodes=int(cfg["n_eval_episodes"]),
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_freq_vec,
        save_path=str(ckpt_dir),
        name_prefix="dapg",
    )

    # ── 训练 ──
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([eval_cb, ckpt_cb]),
        progress_bar=True,
    )

    final_path = ckpt_dir / "final.zip"
    model.save(final_path)
    buffer_path = ckpt_dir / "replay_buffer.pkl"
    model.save_replay_buffer(str(buffer_path))
    print(f"\nfinal model saved → {final_path}")
    print(f"replay buffer saved → {buffer_path}")


if __name__ == "__main__":
    main()
