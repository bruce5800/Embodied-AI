"""bc_pretrain.py — 用 expert demos 监督预训练 SAC actor (Behavior Cloning)。

策略：
  1. 创建 SAC model（actor 随机初始化）
  2. 加载 demos npz
  3. 监督训 actor：MSE(tanh(actor_mean(obs)), expert_action)
  4. critic 不训（保持随机初始化），让 SAC fine-tune 时从零学 critic
     避免 BC 学坏的 critic 误导 fine-tune
  5. 保存整个 SAC model，方便 train_sac.py 用 --load-from 接力

用法：
    python bc_pretrain.py --demos demos/blue_cube_v9.npz \
        --epochs 50 --target blue_cube \
        --out checkpoints/bc_blue_cube
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from env import GraspEnv


REPO_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=str, required=True,
                        help="npz 路径（collect_demos.py 输出）")
    parser.add_argument("--target", default="blue_cube",
                        help="GraspEnv target_object（建模 obs/action space 用）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--obj-radius", type=float, default=None,
                        help="curriculum_radius；BC env 应该用跟 demo 一致的 setup")
    parser.add_argument("--out", type=str, required=True,
                        help="输出目录（保存 policy.zip）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── 1. 加载 demos ──────────────────────────────────────
    demos_path = Path(args.demos)
    print(f"Loading {demos_path}")
    data = np.load(demos_path)
    obs_np = data["obs"]
    action_np = data["action"]
    print(f"  obs shape    : {obs_np.shape}")
    print(f"  action shape : {action_np.shape}")
    print(f"  obs range    : [{obs_np.min():+.3f}, {obs_np.max():+.3f}]")
    print(f"  action range : [{action_np.min():+.3f}, {action_np.max():+.3f}]")

    # 转 torch tensors
    obs_t = torch.from_numpy(obs_np).float()
    action_t = torch.from_numpy(action_np).float()
    n_samples = len(obs_t)

    # ── 2. 创建 SAC model（actor 随机初始化）─────────────────
    target = None if args.target == "random" else args.target
    env = GraspEnv(
        target_object=target,
        max_episode_steps=args.max_episode_steps,
        curriculum_radius=args.obj_radius,
    )

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [256, 256]},
        learning_rate=args.lr,
        buffer_size=10_000,        # BC 不用 buffer
        learning_starts=0,
        verbose=0,
        seed=args.seed,
    )

    actor = model.policy.actor
    device = next(actor.parameters()).device
    obs_t = obs_t.to(device)
    action_t = action_t.to(device)

    # 验证 SAC actor 接口可用
    print(f"\n  device       : {device}")
    print(f"  actor type   : {type(actor).__name__}")

    # ── 3. BC 训练循环 ───────────────────────────────────
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

    print(f"\nBC pretrain: {args.epochs} epochs × {n_samples // args.batch_size} batches")
    print("─" * 60)

    for epoch in range(args.epochs):
        # shuffle 索引
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, args.batch_size):
            batch_idx = perm[i:i + args.batch_size]
            batch_obs = obs_t[batch_idx]
            batch_act = action_t[batch_idx]

            # Max-likelihood BC + sample weighting
            # gripper close 样本（action[5] < 0）加权重 3x，强迫 actor 学到 conditional close
            # 否则 50/50 open/close 分布让 unimodal Gaussian mode collapse 到 open
            mean_actions, log_std, _kwargs = actor.get_action_dist_params(batch_obs)
            actor.action_dist.proba_distribution(mean_actions, log_std)
            expert_clipped = torch.clamp(batch_act, -0.999, 0.999)
            log_prob = actor.action_dist.log_prob(expert_clipped)   # shape (B,)

            # close 样本（gripper expert action < 0）权重 3x
            is_close = batch_act[:, 5] < 0
            weights = torch.where(is_close, 3.0, 1.0)
            loss = -(weights * log_prob).sum() / weights.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch+1:3d}/{args.epochs}: loss = {avg_loss:.5f}")

    # ── 4. 验证（推理一个样本）──────────────────────────────
    actor.eval()
    with torch.no_grad():
        sample_obs = obs_t[:5]
        sample_expert = action_t[:5].cpu().numpy()
        mean, _, _ = actor.get_action_dist_params(sample_obs)
        sample_pred = torch.tanh(mean).cpu().numpy()
        print("\n样本对比（前 5 个 transition）:")
        for i in range(5):
            print(f"  expert  : {sample_expert[i]}")
            print(f"  pred    : {sample_pred[i]}")
            print(f"  diff_norm: {np.linalg.norm(sample_expert[i]-sample_pred[i]):.4f}")
            print()
    actor.train()

    # ── 5. 保存 ─────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "policy.zip"
    model.save(str(model_path))
    print(f"\n✓ 保存 BC pretrained SAC → {model_path}")
    print(f"   后续 fine-tune: python train_sac.py --load-from {model_path} ...")

    env.close()


if __name__ == "__main__":
    main()
