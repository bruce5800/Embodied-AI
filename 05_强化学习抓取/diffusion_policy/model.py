"""Conditional 1D UNet for Diffusion Policy。

Input:  noisy_action_chunk  (B, K, A)   ← treat K (chunk len) as 1D 序列长度
        obs                  (B, O)      ← 28-dim state
        timestep             (B,)        ← integer in [0, T)
Output: predicted_noise      (B, K, A)

Architecture:
  - Sinusoidal timestep embedding → MLP → time_emb (D)
  - obs MLP → obs_emb (D)
  - cond = concat(time_emb, obs_emb) → cond_emb (D)
  - 1D Conv UNet (3 levels) over action sequence (K, A)
  - 每个 ConvBlock 用 FiLM modulation (scale + shift from cond)

参数量目标：~1M（适合 Mac CPU/MPS）
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal time embedding (Vaswani 2017)。
    timesteps: (B,) integer; output (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FiLMConvBlock(nn.Module):
    """1D Conv + GroupNorm + Mish + FiLM (scale + shift from cond)。"""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()  # 原本用 Mish，但 MPS 数值不稳；SiLU 同等效果
        # FiLM: cond → (scale, shift) for out_ch channels
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, K), cond: (B, cond_dim)
        x = self.conv(x)
        x = self.norm(x)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return self.act(x)


class ConditionalUNet1D(nn.Module):
    """1D UNet with FiLM conditioning，paper 同款简化版。

    Channels: 64 → 128 → 256 → 256 (down) → 256 → 128 → 64 (up)
    """

    def __init__(
        self,
        action_dim: int = 6,
        obs_dim: int = 28,
        hidden_dims: tuple = (64, 128, 256),
        cond_dim: int = 256,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        # ── 条件 embedding ─────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_emb_dim = time_emb_dim
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.cond_combine = nn.Linear(time_emb_dim * 2, cond_dim)

        # ── UNet down path ─────────────────────
        self.down_blocks = nn.ModuleList()
        in_ch = action_dim
        self.down_chs = [in_ch] + list(hidden_dims)
        for i in range(len(hidden_dims)):
            self.down_blocks.append(
                FiLMConvBlock(self.down_chs[i], hidden_dims[i], cond_dim)
            )

        # ── 中间 ──
        self.mid_block = FiLMConvBlock(hidden_dims[-1], hidden_dims[-1], cond_dim)

        # ── up path（with skip concat） ─────────
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(hidden_dims))):
            in_skip = hidden_dims[i] * 2  # concat skip
            out = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            self.up_blocks.append(FiLMConvBlock(in_skip, out, cond_dim))

        # ── 输出层 ──
        self.out_conv = nn.Conv1d(hidden_dims[0], action_dim, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,         # (B, K, A) noisy actions
        obs: torch.Tensor,        # (B, O)
        timesteps: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        # 1. cond = combine(time_emb, obs_emb)
        t_emb = sinusoidal_embedding(timesteps, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)                     # (B, T)
        o_emb = self.obs_proj(obs)                       # (B, T)
        cond = self.cond_combine(torch.cat([t_emb, o_emb], dim=-1))  # (B, cond_dim)

        # 2. transpose for Conv1d: (B, K, A) → (B, A, K)
        h = x.transpose(1, 2)

        # 3. down path（保存 skip）
        skips = []
        for block in self.down_blocks:
            h = block(h, cond)
            skips.append(h)

        # 4. mid
        h = self.mid_block(h, cond)

        # 5. up path（concat skip from down）
        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        # 6. 输出 → predicted noise (B, A, K) → (B, K, A)
        out = self.out_conv(h)
        return out.transpose(1, 2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick sanity test
    model = ConditionalUNet1D(action_dim=6, obs_dim=28)
    print(f"params: {count_params(model):,}")
    x = torch.randn(4, 8, 6)
    obs = torch.randn(4, 28)
    t = torch.randint(0, 100, (4,))
    out = model(x, obs, t)
    print(f"input  x:  {x.shape}")
    print(f"input  obs:{obs.shape}")
    print(f"output:    {out.shape}")
