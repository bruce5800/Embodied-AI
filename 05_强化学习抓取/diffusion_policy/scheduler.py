"""DDIM noise scheduler。

Forward: a_k = sqrt(α̅_k) · a_0 + sqrt(1-α̅_k) · ε
Reverse (DDIM): a_{k-1} = sqrt(α̅_{k-1}) · pred_a_0 + sqrt(1-α̅_{k-1}) · pred_noise

squaredcos_cap_v2 schedule（paper 默认）：β_k = 1 - α̅_k / α̅_{k-1}，
其中 α̅_k = cos²((k/T + s) / (1+s) · π/2)

DDIM 加速：训练用 100 步，推理用 10 步（subset of timesteps）。
"""

from __future__ import annotations

import math
import torch


def _squaredcos_cap_v2(num_steps: int) -> torch.Tensor:
    """Improved DDPM noise schedule (Nichol & Dhariwal 2021)。
    返回 alphas_cumprod (T,)，递减从 ~1.0 到 ~0.0。
    """
    s = 0.008
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps) / num_steps
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = betas.clamp(max=0.999)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


class DDIMScheduler:
    """DDPM-style training，DDIM-style inference 加速。"""

    def __init__(self, num_train_steps: int = 100, device: str = "cpu"):
        self.num_train_steps = num_train_steps
        self.device = device

        alphas_cumprod = _squaredcos_cap_v2(num_train_steps).to(device)
        self.alphas_cumprod = alphas_cumprod                       # ᾱ_k
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward 加噪：a_k = sqrt(ᾱ_k)·a_0 + sqrt(1-ᾱ_k)·ε"""
        # broadcast timestep 到 (B, 1, 1) 方便跟 (B, K, A) 相乘
        s_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        s_1ma = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        return s_alpha * original + s_1ma * noise

    def set_inference_steps(self, num_inference_steps: int = 10):
        """选 num_inference_steps 个均匀分布的 timestep 做推理。"""
        # e.g., 训练 100, 推理 10 → [99, 89, 79, ..., 9]
        step = self.num_train_steps // num_inference_steps
        self.inference_timesteps = torch.arange(
            self.num_train_steps - 1, -1, -step, device=self.device
        )[:num_inference_steps]

    @torch.no_grad()
    def step(
        self,
        pred_noise: torch.Tensor,
        timestep_idx: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """单步 DDIM reverse update。
        timestep_idx: inference_timesteps 中的索引（0 到 num_inference_steps-1）
        """
        t = int(self.inference_timesteps[timestep_idx])
        prev_t = int(self.inference_timesteps[timestep_idx + 1]) \
                 if timestep_idx + 1 < len(self.inference_timesteps) else -1

        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=self.device)

        # 从 noisy sample 反推 pred_a_0
        pred_a_0 = (sample - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()

        # 关键：在 t 大时 α̅_t≈0，除法会让 pred_a_0 爆到几百。
        # 真实 a_0 已 normalize 到 [-1, 1]，clip 后再重组。
        # 这是 HuggingFace DDIMScheduler 默认 clip_sample=True 做的事，漏了 → 推理发散。
        pred_a_0 = pred_a_0.clamp(-1.0, 1.0)

        # 重组 a_{prev_t}
        prev_sample = alpha_prev.sqrt() * pred_a_0 + (1 - alpha_prev).sqrt() * pred_noise
        return prev_sample
