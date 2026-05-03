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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import GraspEnv


REPO_ROOT = Path(__file__).resolve().parent


def make_env(target_object, max_episode_steps, seed, curriculum_radius=None):
    """构造单个 GraspEnv（包 Monitor 用于 EvalCallback）— 给 eval / 单 env 训练用。"""
    env = GraspEnv(
        target_object=target_object,
        max_episode_steps=max_episode_steps,
        curriculum_radius=curriculum_radius,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_env_fn(target_object, max_episode_steps, seed, curriculum_radius=None):
    """工厂函数（闭包不捕获 mujoco 句柄，给 SubprocVecEnv 用）。"""
    def _init():
        env = GraspEnv(
            target_object=target_object,
            max_episode_steps=max_episode_steps,
            curriculum_radius=curriculum_radius,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def make_train_vec_env(target, max_episode_steps, seed, n_envs, curriculum_radius=None):
    """n_envs > 1 用 SubprocVecEnv（多进程），= 1 用 DummyVecEnv（单进程）。"""
    fns = [make_env_fn(target, max_episode_steps, seed=seed + i,
                       curriculum_radius=curriculum_radius)
           for i in range(n_envs)]
    if n_envs > 1:
        return SubprocVecEnv(fns, start_method="spawn")
    return DummyVecEnv(fns)


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
    parser.add_argument("--n-envs", type=int, default=1,
                        help="并行训练 env 数量（SubprocVecEnv）。"
                             "M 系列 Mac 推荐 4-6。eval env 始终单进程。")
    parser.add_argument("--obj-radius", type=float, default=None,
                        help="反向课程：target 物体在 zone ±radius 方框内随机。"
                             "None=全场景。建议阶梯 0.05 → 0.15 → None。")
    parser.add_argument("--load-from", type=str, default=None,
                        help="从已有 ckpt 继续训练（路径到 .zip 文件）。"
                             "也可以是 BC pretrain 的 policy.zip。")
    parser.add_argument("--load-demos", type=str, default=None,
                        help="npz 路径（collect_demos.py 输出），把 demo transitions "
                             "灌入 replay buffer。配合 --load-from 是 SACfD 启动方式。")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖 config 的 learning_rate。BC fine-tune 推荐 3e-5（小 10x）"
                             "防 critic 漂移把 actor 洗掉。")
    parser.add_argument("--ent-coef", type=str, default=None,
                        help="覆盖 ent_coef（'auto' / 0.05 / 0.01）。BC fine-tune 推荐"
                             "更小 (0.05) 让 actor 更 deterministic 接近 BC")
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
    print(f"  n_envs          : {args.n_envs}  "
          f"({'SubprocVecEnv' if args.n_envs > 1 else 'DummyVecEnv'})")
    print(f"  obj_radius      : {args.obj_radius}  "
          f"(None=full table)")
    print(f"  load_from       : {args.load_from}")
    print(f"  log_dir         : {log_dir}")
    print(f"  ckpt_dir        : {ckpt_dir}")
    print("─" * 60)

    # ── env ──
    train_env = make_train_vec_env(
        target, args.max_episode_steps, args.seed, args.n_envs,
        curriculum_radius=args.obj_radius,
    )
    # eval env 单进程足够（n_eval_episodes=20 串行也很快）
    # 关键：eval env 也用同样的 curriculum_radius，否则训练/评估指标不可比
    eval_env = make_env(target, args.max_episode_steps,
                        seed=args.seed + 10_000,
                        curriculum_radius=args.obj_radius)

    # ── 模型 ──
    # VecEnv 下 train_freq=1 含义是"每 vec_step 训 1 次"（= 收集 n_envs 个 transitions），
    # 直接用 cfg 的 gradient_steps（不补偿乘 n_envs），换 wall-clock 加速。
    # SAC 文献验证 update-to-data ratio 0.1 ~ 1.0 都能学，N=8 + grad=1 足够。
    grad_steps = int(cfg["gradient_steps"])

    # LR linear decay：缓解长训后期 critic 漂移导致的 reward 退步
    # fine-tune 时（load_from 给定）用 1/3 的 LR 起步，避免洗掉已学策略
    if args.lr is not None:
        base_lr = float(args.lr)
        print(f"  → manual lr override: {base_lr:.2e}")
    else:
        base_lr = float(cfg["learning_rate"])
        if args.load_from is not None:
            base_lr = base_lr / 3.0
            print(f"  → fine-tune mode: lr scaled to {base_lr:.2e}")
    def lr_schedule(progress_remaining: float) -> float:
        # SB3 传入 1.0 → 0.0；最低保留 10% lr
        return base_lr * max(progress_remaining, 0.1)

    if args.load_from is not None:
        # 继续训练：加载 ckpt + 替换 env（新 curriculum_radius）+ 替换 lr_schedule
        print(f"  → loading model from {args.load_from}")
        model = SAC.load(
            args.load_from,
            env=train_env,
            tensorboard_log=str(log_dir),
            custom_objects={"learning_rate": lr_schedule},
        )
        # 加载 replay buffer（如果存在）—— 防止 fine-tune 时 critic 从空 buffer 学崩
        from pathlib import Path as _P
        buffer_path = _P(args.load_from).with_name("replay_buffer.pkl")
        if buffer_path.exists():
            print(f"  → loading replay buffer from {buffer_path}")
            model.load_replay_buffer(str(buffer_path))
        else:
            print(f"  ⚠ no replay buffer at {buffer_path} "
                  f"(BC ckpt 通常无 buffer，配合 --load-demos)")
        model.learning_starts = 0
    else:
        ent_coef_value = args.ent_coef if args.ent_coef is not None else cfg["ent_coef"]
        # 解析数字（不是 "auto"）
        try:
            ent_coef_value = float(ent_coef_value)
        except (ValueError, TypeError):
            pass  # 保留 "auto" 字符串
        model = SAC(
            cfg["policy"],
            train_env,
            learning_rate=lr_schedule,
            buffer_size=int(cfg["buffer_size"]),
            batch_size=int(cfg["batch_size"]),
            tau=float(cfg["tau"]),
            gamma=float(cfg["gamma"]),
            ent_coef=ent_coef_value,
            target_entropy=cfg["target_entropy"],
            learning_starts=int(cfg["learning_starts"]),
            gradient_steps=grad_steps,
            train_freq=int(cfg["train_freq"]),
            policy_kwargs=cfg.get("policy_kwargs", {}),
            tensorboard_log=str(log_dir),
            seed=args.seed,
            verbose=1,
        )

    # ── 灌 demo transitions 进 replay buffer (SACfD style) ──
    if args.load_demos is not None:
        from pathlib import Path as _P
        demos_path = _P(args.load_demos)
        print(f"  → loading demos from {demos_path}")
        data = np.load(demos_path)
        d_obs = data["obs"]
        d_act = data["action"]
        d_rew = data["reward"]
        d_nobs = data["next_obs"]
        d_done = data["done"]
        n_demo = len(d_obs)

        # SB3 ReplayBuffer.add 期望 shape=(n_envs, ...)。我们的 demo 是单序列，
        # 按 vec batch 大小 (n_envs) 切片 add。剩余不到 n_envs 个的丢弃（< 8 transitions 损失忽略不计）
        n_envs = model.replay_buffer.n_envs
        n_full = (n_demo // n_envs) * n_envs
        n_added = 0
        for start in range(0, n_full, n_envs):
            end = start + n_envs
            obs_b = d_obs[start:end]
            nobs_b = d_nobs[start:end]
            act_b = d_act[start:end]
            rew_b = d_rew[start:end].astype(np.float32)
            done_b = d_done[start:end].astype(np.float32)
            model.replay_buffer.add(
                obs=obs_b,
                next_obs=nobs_b,
                action=act_b,
                reward=rew_b,
                done=done_b,
                infos=[{} for _ in range(n_envs)],
            )
            n_added += n_envs
        skipped = n_demo - n_added
        print(f"  → loaded {n_added:,} demo transitions into replay buffer "
              f"(skipped {skipped} tail; n_envs={n_envs})")
        model.learning_starts = 0   # buffer 有数据，不要 random rollout

    # ── 回调 ──
    # VecEnv 下 callback 的 freq 是 vec_steps，要除以 n_envs 才是 env_steps 节奏
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
    # 同时保存 replay buffer 给后续 fine-tune 用
    buffer_path = ckpt_dir / "replay_buffer.pkl"
    model.save_replay_buffer(str(buffer_path))
    print(f"\nfinal model saved → {final_path}")
    print(f"replay buffer saved → {buffer_path}")

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
