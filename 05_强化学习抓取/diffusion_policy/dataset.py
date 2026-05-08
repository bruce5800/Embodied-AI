"""把 collect_demos.py 输出的 npz 转成 (obs, action_chunk) 训练样本。

数据流:
  npz: { obs (N, 28), action (N, 6), reward (N,), next_obs (N, 28), done (N,) }
   ↓
  按 done flag 切成 trajectories
   ↓
  每个 trajectory 内 sliding window 出 (obs_t, action_chunk_{t..t+K-1})
  trajectory 末尾不足 K 步：用最后 action 重复填充
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DiffusionDataset(Dataset):
    """从 demo npz 生成 (obs, action_chunk) pairs。"""

    def __init__(
        self,
        npz_path: str | Path,
        horizon: int = 8,
        pad_last: bool = True,
    ):
        data = np.load(npz_path)
        obs_all = data["obs"].astype(np.float32)        # (N, 28)
        action_all = data["action"].astype(np.float32)  # (N, 6)
        done_all = data["done"].astype(np.bool_)        # (N,)

        self.horizon = horizon

        # 按 done 切 trajectory
        # trajectories: list of (obs_traj, action_traj) where each is (T_i, dim)
        trajectories = []
        start = 0
        for i, done in enumerate(done_all):
            if done:
                trajectories.append((obs_all[start:i + 1], action_all[start:i + 1]))
                start = i + 1
        if start < len(obs_all):  # 最后一段没 done flag 也保留
            trajectories.append((obs_all[start:], action_all[start:]))

        # 每个 trajectory 内 sliding window 出 (obs_t, action_chunk)
        self.samples_obs = []
        self.samples_action = []
        for obs_traj, action_traj in trajectories:
            T = len(obs_traj)
            for t in range(T):
                # action_chunk: action[t : t+K]，不够就 pad 最后 action
                end = t + horizon
                if end <= T:
                    chunk = action_traj[t:end]
                elif pad_last:
                    n_pad = end - T
                    chunk = np.concatenate(
                        [action_traj[t:], np.tile(action_traj[-1:], (n_pad, 1))],
                        axis=0,
                    )
                else:
                    continue   # 丢弃尾部
                assert chunk.shape == (horizon, action_traj.shape[1])
                self.samples_obs.append(obs_traj[t])
                self.samples_action.append(chunk)

        self.samples_obs = np.stack(self.samples_obs)         # (M, 28)
        self.samples_action = np.stack(self.samples_action)   # (M, K, 6)

        print(f"[DiffusionDataset] loaded {len(trajectories)} trajectories")
        print(f"  total transitions: {len(obs_all):,}")
        print(f"  samples: {len(self.samples_obs):,}")
        print(f"  obs shape:    {self.samples_obs.shape}")
        print(f"  action shape: {self.samples_action.shape}")

    def __len__(self) -> int:
        return len(self.samples_obs)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.samples_obs[idx]),
            torch.from_numpy(self.samples_action[idx]),
        )

    @property
    def action_dim(self) -> int:
        return self.samples_action.shape[-1]

    @property
    def obs_dim(self) -> int:
        return self.samples_obs.shape[-1]


def make_dataloader(
    npz_path: str | Path,
    horizon: int = 8,
    batch_size: int = 256,
    num_workers: int = 0,
    shuffle: bool = True,
) -> tuple[DataLoader, DiffusionDataset]:
    ds = DiffusionDataset(npz_path, horizon=horizon)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return dl, ds


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "demos/blue_cube_v10_strict.npz"
    dl, ds = make_dataloader(path)
    print(f"\nbatches/epoch: {len(dl)}")
    for obs, action in dl:
        print(f"obs batch:    {obs.shape} dtype={obs.dtype}")
        print(f"action batch: {action.shape} dtype={action.dtype}")
        print(f"obs stats:    [{obs.min():+.2f}, {obs.max():+.2f}]")
        print(f"action stats: [{action.min():+.2f}, {action.max():+.2f}]")
        break
