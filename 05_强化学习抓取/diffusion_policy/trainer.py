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
    # 关键：先在 CPU init（保证默认 init 数值稳定），再 .to(mps)
    # 不加 seed + 直接 to(mps) 在我们 model 上偶发出 NaN init（torch MPS bug）
    torch.manual_seed(0)
    model = ConditionalUNet1D(action_dim=action_dim, obs_dim=obs_dim)
    # 检查 init weights
    bad = [n for n, p in model.named_parameters() if torch.isnan(p).any().item() or torch.isinf(p).any().item()]
    if bad:
        raise RuntimeError(f"NaN in CPU init weights: {bad}")
    model = model.to(device)
    # 再次 check（to mps 后）
    bad = [n for n, p in model.named_parameters() if torch.isnan(p).any().item() or torch.isinf(p).any().item()]
    if bad:
        print(f"⚠ NaN after .to({device}): {bad[:3]}; falling back to cpu")
        device = torch.device("cpu")
        model = model.cpu()
    ema = EMAModel(model, decay=ema_decay)
    scheduler = DDIMScheduler(num_train_steps=num_train_steps, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # LR schedule: linear warmup (前 500 step) → cosine
    # 第一 epoch 不 warmup 时 loss 爆到 1e22 (MPS 数值不稳)
    total_steps = epochs * len(dataloader)
    warmup_steps = min(500, total_steps // 10)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps,
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
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
            # 注意：MPS 上 non_blocking=True 会读到未完成 copy 的垃圾内存 → NaN
            # CPU 上 non_blocking 也无意义（同一 host），所以一律 False
            obs = obs.to(device)
            action_chunk = action_chunk.to(device)
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

            loss_val = loss.item()
            if not (loss_val == loss_val):   # NaN check
                print(f"!! NaN detected at epoch {epoch} batch {n_batches}")
                # 检查 model weights
                nan_w = [n for n, p in model.named_parameters() if torch.isnan(p).any().item()]
                print(f"   nan model weights: {len(nan_w)} (first: {nan_w[:2]})")
                nan_ema = [n for n, p in ema.ema_model.named_parameters() if torch.isnan(p).any().item()]
                print(f"   nan ema weights:   {len(nan_ema)}")
                raise RuntimeError("NaN in training")
            epoch_loss += loss_val
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
