"""DiffusionPolicy 推理 wrapper。

- 加载 ema.pt
- predict_action(obs) → 调 DDIM 反向去噪生成 action chunk (K, A)
- receding horizon: 每次执行 N 步后重新预测
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import ConditionalUNet1D
from .scheduler import DDIMScheduler


class DiffusionPolicy:
    def __init__(
        self,
        ckpt_path: str | Path,
        device: str | torch.device = "cpu",
        num_inference_steps: int = 10,
        n_action_steps: int = 4,
    ):
        """加载 ema ckpt + 配置 inference scheduler。

        Args:
            n_action_steps: receding horizon - 每次预测后执行多少步。
                          推理 K=8 步 chunk，执行 4 步，剩下 overlap 用于平滑。
        """
        device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.obs_dim = ckpt["obs_dim"]
        self.action_dim = ckpt["action_dim"]
        self.horizon = ckpt["horizon"]
        self.num_train_steps = ckpt["num_train_steps"]
        self.n_action_steps = n_action_steps
        self.device = device

        self.model = ConditionalUNet1D(
            action_dim=self.action_dim, obs_dim=self.obs_dim
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.scheduler = DDIMScheduler(
            num_train_steps=self.num_train_steps, device=device
        )
        self.scheduler.set_inference_steps(num_inference_steps)
        self.num_inference_steps = num_inference_steps

        # Receding horizon state
        self._chunk: np.ndarray | None = None      # 缓存的 action chunk (K, A)
        self._step_in_chunk: int = 0               # 当前在 chunk 里走到第几步

    @torch.no_grad()
    def _predict_chunk(self, obs: np.ndarray) -> np.ndarray:
        """从 obs 生成一个 action chunk (K, A)，DDIM K_inf 步。"""
        obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)  # (1, O)

        # 从纯噪声开始
        a = torch.randn(1, self.horizon, self.action_dim, device=self.device)

        # DDIM 反向去噪
        for i, t in enumerate(self.scheduler.inference_timesteps):
            timestep_batch = torch.full((1,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(a, obs_t, timestep_batch)
            a = self.scheduler.step(pred_noise, i, a)

        chunk = a.squeeze(0).cpu().numpy()       # (K, A)
        # 安全裁剪到 [-1, 1]
        return np.clip(chunk, -1.0, 1.0)

    def reset(self):
        """新 episode 开始，清空缓存。"""
        self._chunk = None
        self._step_in_chunk = 0

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """对外接口：每次调用返回 1 个 action (A,)。

        Receding horizon 逻辑：
          - 第 1 次调用：predict chunk, 取第 0 步
          - 接下来 n_action_steps-1 次：取 chunk[1..N-1]
          - 第 n_action_steps 次：重新 predict chunk, 取第 0 步
        """
        # 注意 deterministic 在 diffusion 里不直接生效（采样必有 noise），
        # 但 inference 不用 sample noise（pred_noise 是 deterministic forward）
        if self._chunk is None or self._step_in_chunk >= self.n_action_steps:
            self._chunk = self._predict_chunk(obs)
            self._step_in_chunk = 0

        action = self._chunk[self._step_in_chunk]
        self._step_in_chunk += 1
        return action.astype(np.float32)
