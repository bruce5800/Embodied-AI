"""train_diffusion.py — Diffusion Policy 训练入口。

用法：
    # smoke test (5 epoch CPU)
    python train_diffusion.py --epochs 5 --device cpu --run-name dp_smoke

    # 正式（150 epoch on MPS, ~30-60 min）
    python train_diffusion.py --epochs 150 --device mps --run-name dp_blue_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusion_policy import make_dataloader, train_diffusion_policy


REPO_ROOT = Path(__file__).resolve().parent


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", default="demos/blue_cube_v10_strict.npz")
    parser.add_argument("--horizon", type=int, default=8,
                        help="Action chunk 长度 (paper K=8)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-train-steps", type=int, default=100,
                        help="Diffusion 时间步数（DDIM 推理时只用 10 步）")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--device", default=auto_device(),
                        help="cpu / mps / cuda")
    parser.add_argument("--run-name", default="dp_blue_v1")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (mps 推荐 0)")
    args = parser.parse_args()

    save_dir = REPO_ROOT / args.ckpt_dir / args.run_name

    print("─" * 60)
    print(f"  demos       : {args.demos}")
    print(f"  device      : {args.device}")
    print(f"  horizon     : {args.horizon}")
    print(f"  epochs      : {args.epochs}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  lr          : {args.lr}")
    print(f"  save_dir    : {save_dir}")
    print("─" * 60)

    dl, ds = make_dataloader(
        REPO_ROOT / args.demos,
        horizon=args.horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_diffusion_policy(
        dataloader=dl,
        obs_dim=ds.obs_dim,
        action_dim=ds.action_dim,
        horizon=args.horizon,
        num_train_steps=args.num_train_steps,
        epochs=args.epochs,
        lr=args.lr,
        ema_decay=args.ema_decay,
        device=args.device,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
