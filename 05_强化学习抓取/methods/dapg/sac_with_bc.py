"""SACWithBC — SAC + Behavior Cloning regularization on actor (DAPG-style).

核心修改：
  - actor loss = SAC_actor_loss + λ * BC_loss
  - BC_loss = -log π(a_demo | s_demo)
  - λ 随训练步数衰减（初始大、最终小）
  - close action 加权（避免 mode collapse 到"永远开爪"）

为什么这样能防止 SACfD 的 actor 漂移？
  - 标准 SACfD：actor 跟随 critic 信号，但 early critic 估值不稳 → actor 漂走
  - DAPG：actor 永远有 BC anchor，即使 critic 错也不会跑太远
  - λ 衰减让 RL 在后期主导，actor 能超越 expert（如果 critic 学到了）

继承关系：直接继承 SB3 SAC，override train() 方法。
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer


class SACWithBC(SAC):
    """SAC + actor-side BC regularization.

    Args added beyond SAC:
        bc_lambda_init: 初始 BC reg 权重（推荐 1.0）
        bc_lambda_decay: 每个 gradient step 衰减率（推荐 0.9999 慢衰，0.999 快衰）
        bc_lambda_min: λ 最小值（保留底线 BC anchor）
        bc_close_weight: close action 样本权重（解 mode collapse）
        bc_batch_size: BC update 的 batch size（None = 跟 SAC batch_size 同）
    """

    def __init__(
        self,
        *args,
        bc_lambda_init: float = 1.0,
        bc_lambda_decay: float = 0.9999,
        bc_lambda_min: float = 0.05,
        bc_close_weight: float = 3.0,
        bc_batch_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bc_lambda_init = float(bc_lambda_init)
        self.bc_lambda = float(bc_lambda_init)
        self.bc_lambda_decay = float(bc_lambda_decay)
        self.bc_lambda_min = float(bc_lambda_min)
        self.bc_close_weight = float(bc_close_weight)
        self.bc_batch_size = bc_batch_size
        self.demo_buffer: Optional[ReplayBuffer] = None

    def set_demo_buffer(self, demo_buffer: ReplayBuffer):
        """注入预加载的 demo buffer。"""
        self.demo_buffer = demo_buffer

    def load_demos_from_npz(self, npz_path: str):
        """从 collect_demos.py 输出的 npz 加载 demos 到 demo_buffer。"""
        data = np.load(npz_path)
        n_demo = len(data["obs"])
        self.demo_buffer = ReplayBuffer(
            buffer_size=n_demo + 100,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=1,
        )
        for i in range(n_demo):
            self.demo_buffer.add(
                obs=data["obs"][i:i+1],
                next_obs=data["next_obs"][i:i+1],
                action=data["action"][i:i+1],
                reward=np.array([data["reward"][i]], dtype=np.float32),
                done=np.array([data["done"][i]], dtype=np.float32),
                infos=[{}],
            )
        print(f"  → SACWithBC.demo_buffer loaded {n_demo:,} transitions")

    def _compute_bc_loss(self, batch_size: int) -> torch.Tensor:
        """从 demo_buffer 采样，计算 weighted BC loss（log-likelihood）。"""
        if self.demo_buffer is None or self.demo_buffer.size() == 0:
            return torch.tensor(0.0, device=self.device)

        bs = self.bc_batch_size or batch_size
        bs = min(bs, int(self.demo_buffer.size()))
        demo = self.demo_buffer.sample(bs, env=self._vec_normalize_env)

        # actor distribution params
        mean_actions, log_std, _ = self.actor.get_action_dist_params(demo.observations)
        self.actor.action_dist.proba_distribution(mean_actions, log_std)

        # log_prob on demo actions (squashed [-1, 1] format)
        expert_clipped = torch.clamp(demo.actions, -0.999, 0.999)
        log_prob = self.actor.action_dist.log_prob(expert_clipped)   # (B,)

        # close action weighting (action[5] < 0 → close)
        is_close = demo.actions[:, 5] < 0.0
        weights = torch.where(
            is_close,
            torch.tensor(self.bc_close_weight, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        bc_loss = -(weights * log_prob).sum() / weights.sum()
        return bc_loss

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """完整 override SB3 SAC train()，actor loss 加 BC reg。"""
        # set actor / critic to train mode
        self.policy.set_training_mode(True)
        # ent_coef optimizer
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, bc_losses = [], [], []

        for _ in range(gradient_steps):
            # 采样 replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)

            # === ent coef update ===
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            # === critic update ===
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(cq, target_q_values) for cq in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # === actor update with BC reg ===
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            min_qf_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(min_qf_pi, dim=1, keepdim=True)
            sac_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # BC regularization
            bc_loss = self._compute_bc_loss(batch_size)
            actor_loss = sac_actor_loss + self.bc_lambda * bc_loss

            actor_losses.append(actor_loss.item())
            bc_losses.append(bc_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # === target network soft update ===
            from stable_baselines3.common.utils import polyak_update
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            # === BC lambda 衰减 ===
            self.bc_lambda = max(
                self.bc_lambda * self.bc_lambda_decay, self.bc_lambda_min)

        self._n_updates += gradient_steps
        # logging
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/bc_lambda", self.bc_lambda)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
