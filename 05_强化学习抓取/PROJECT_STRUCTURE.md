# 项目结构

按算法路线 + 用途分目录组织。每个脚本头部 `sys.path.insert` 让 `from env import` 全局可用。

```
05_强化学习抓取/
│
├── env/                          # GraspEnv 环境（所有方法共用）
│   ├── grasp_env.py              # gym.Env 主体（28-dim obs / 6-dim joint action）
│   ├── scene_constants.py        # 物体 / zone / 关节常量
│   ├── randomization.py          # 物体随机化（含课程化 target_override）
│   └── reward_shaping.py         # phase-gated dense reward
│
├── diffusion_policy/             # Diffusion Policy 模块（self-contained 库）
│   ├── scheduler.py              # DDIM noise schedule
│   ├── model.py                  # Conditional 1D UNet (1.2M params)
│   ├── dataset.py                # (obs, action_chunk) 数据加载
│   ├── trainer.py                # 训练 loop + EMA
│   └── policy.py                 # DDIM 推理 + receding horizon
│
├── methods/                      # 各算法路线
│   │
│   ├── sac/                      # 路线 A: SAC + 反向课程
│   │   ├── train.py
│   │   └── configs/sac.yaml
│   │
│   ├── expert/                   # 路线 B: IK 脚本式 expert + demo 收集
│   │   ├── expert.py
│   │   └── collect_demos.py
│   │
│   ├── dapg/                     # 路线 C: BC + DAPG
│   │   ├── bc_pretrain.py
│   │   ├── sac_with_bc.py
│   │   └── train_dapg.py
│   │
│   └── diffusion/                # 路线 D: Diffusion Policy 入口
│       ├── train.py
│       └── eval.py
│
├── tools/                        # 通用工具
│   ├── diagnose.py               # 严格诊断（含 stably_held + placed_via_held）
│   ├── eval.py                   # SB3 模型通用评估（SAC/DAPG/BC）
│   ├── eval_final.py             # 综合评估
│   ├── record_real_success.py    # 录"真抓-放"成功视频
│   ├── record_push_hack.py       # 录"假阳性"反例视频
│   └── smoke_test.py             # env sanity check
│
├── checkpoints/
│   ├── sac/
│   │   ├── m1_yellow_v10_600k/   # SAC stage 1 (curriculum r=0.05)
│   │   └── m1_yellow_s2_v3_buf/  # ⭐ SAC stage 2 真实最佳 (placed_via_held 34%)
│   ├── bc/
│   │   └── bc_blue_cube_v4/      # BC 失败案例（保留作 negative result 文档）
│   ├── dapg/
│   │   └── m1_dapg_v2/           # DAPG 失败案例
│   └── diffusion/
│       └── dp_blue_v1/           # Diffusion Policy（150 epoch CPU）
│
├── demos/                        # expert 收集的 demo npz
│   └── blue_cube_v10_strict.npz  # 155 traj / 30k transitions
│
├── demos_videos/                 # 演示视频（成果展示）
├── eval_renders/                 # eval 临时视频
├── logs/                         # tensorboard 日志
│
├── README.md
├── 实验日志.md                   # v1-v11 SAC + v1-v9 expert + BC + DAPG + Diffusion 全过程
└── PROJECT_STRUCTURE.md          # 本文件
```

## 调用示例

```bash
# 路线 A: SAC 训练
python methods/sac/train.py --target yellow_cylinder --total-timesteps 600000 --n-envs 8 \
    --obj-radius 0.05 --run-name m1_yellow

# 路线 B: 收集 expert demos（4-worker 并行）
python methods/expert/collect_demos.py --target blue_cube --n-success 50 --n-workers 4

# 路线 C: BC pretrain + DAPG
python methods/dapg/bc_pretrain.py --demos demos/blue_cube_v10_strict.npz \
    --target blue_cube --epochs 50 --out checkpoints/bc/bc_v4
python methods/dapg/train_dapg.py --demos demos/blue_cube_v10_strict.npz \
    --target blue_cube --total-timesteps 200000 --n-envs 8 --soften-contacts \
    --run-name m1_dapg_v2

# 路线 D: Diffusion Policy
python methods/diffusion/train.py --epochs 150 --device cpu --run-name dp_blue_v1
python methods/diffusion/eval.py --ckpt checkpoints/diffusion/dp_blue_v1/ema.pt \
    --target blue_cube --episodes 50 --soften-contacts

# 通用工具
python tools/diagnose.py --ckpt checkpoints/sac/m1_yellow_s2_v3_buf/best_model.zip \
    --target yellow_cylinder --obj-radius 0.10 --episodes 50
python tools/smoke_test.py                  # env sanity check
python tools/record_real_success.py --ckpt <ckpt> --target <obj>
```

## Import 解决方案

每个 `methods/X/Y.py` 和 `tools/X.py` 头部都 inject 了 sys.path：

```python
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))   # methods/X/Y.py
# 或 parents[1]                                                  # tools/X.py
```

这让 `from env import GraspEnv` / `from diffusion_policy import ...` 在任何地方都 work，不需要 pip install -e .。
