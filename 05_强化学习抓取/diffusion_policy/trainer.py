"""训练 loop + EMA weights。

EMA: 推理时用平滑后的权重（θ_ema = 0.999·θ_ema + 0.001·θ_train），
减少训练 noise，paper 实测能提升 ~5-10% success rate。
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from .model import ConditionalUNet1D, count_params
from .scheduler import DDIMScheduler


class EMAModel:
    """指数滑动平均权重 (Exponential Moving Average)。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


def train_diffusion_policy(
    dataloader,
    obs_dim: int,
    action_dim: int,
    horizon: int = 8,
    *,
    num_train_steps: int = 100,
    epochs: int = 150,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    ema_decay: float = 0.999,
    device: str | torch.device = "cpu",
    save_dir: str | Path = "checkpoints/diffusion_v1",
    log_every: int = 5,
):
    """完整训练 loop。返回 model + ema_model。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    model = ConditionalUNet1D(action_dim=action_dim, obs_dim=obs_dim).to(device)
    ema = EMAModel(model, decay=ema_decay)
    scheduler = DDIMScheduler(num_train_steps=num_train_steps, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # cosine LR schedule
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader)
    )

    print(f"[trainer] device: {device}")
    print(f"[trainer] params: {count_params(model):,}")
    print(f"[trainer] epochs: {epochs}, batches/epoch: {len(dataloader)}")
    print(f"[trainer] num_train_steps: {num_train_steps}, ema_decay: {ema_decay}")
    print("─" * 60)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for obs, action_chunk in dataloader:
            obs = obs.to(device, non_blocking=True)
            action_chunk = action_chunk.to(device, non_blocking=True)
            B = action_chunk.shape[0]

            # 1. 采样 timestep
            t = torch.randint(0, num_train_steps, (B,), device=device)
            # 2. forward 加噪
            noise = torch.randn_like(action_chunk)
            noisy = scheduler.add_noise(action_chunk, noise, t)
            # 3. 模型预测噪声
            pred_noise = model(noisy, obs, t)
            # 4. MSE loss (在 noise 上)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            # 5. 梯度 step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            # 6. EMA update
            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if epoch == 0 or (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:3d}/{epochs}: "
                  f"loss = {avg_loss:.5f}  lr = {cur_lr:.2e}")

    # 保存最终 ckpt
    model_path = save_dir / "model.pt"
    ema_path = save_dir / "ema.pt"
    torch.save({
        "model_state": model.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "horizon": horizon,
        "num_train_steps": num_train_steps,
    }, model_path)
    torch.save({
        "model_state": ema.ema_model.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "horizon": horizon,
        "num_train_steps": num_train_steps,
    }, ema_path)
    print(f"\n  ✓ saved model → {model_path}")
    print(f"  ✓ saved ema   → {ema_path}")

    return model, ema.ema_model, losses
