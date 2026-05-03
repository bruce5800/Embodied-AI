# 05 强化学习抓取 — SAC + Curriculum + BC

在 04 阶段（脚本式抓取 + LLM 任务规划）之上，探索两条 RL 路线：
1. **SAC + 反向课程学习**：从零训 RL，达成 placed 47%（受限场景）
2. **BC + Expert demo**：用脚本式 expert 收集 demo 训行为克隆，达成 lift 80%（最难场景）

## 当前最佳成果

### 两条路线对比

| 路线 | 最佳 ckpt | 评估场景 | placed | lift |
|------|---------|---------|--------|------|
| **SAC + 课程化** | `checkpoints/m1_yellow_s2_v3_buf/best_model.zip` | yellow_cylinder, r=0.10 | **47%** | 53% |
| 同上 | 同 | yellow_cylinder, full table | 9% | 23% |
| **BC alone (50 demos)** | `checkpoints/bc_blue_cube/policy.zip` | blue_cube, full table | 0% | **80%** |
| BC + SACfD（fine-tune 失败） | _-_ | _-_ | 0% | 0% |

### 演示视频（demos_videos/）

| 视频 | 内容 |
|------|------|
| [bc_alone_blue_cube_lift1.mp4](demos_videos/bc_alone_blue_cube_lift1.mp4) | BC actor 在全场景下抓 cube |
| [bc_alone_blue_cube_lift2.mp4](demos_videos/bc_alone_blue_cube_lift2.mp4) | 另一角度成功 lift |
| [bc_alone_blue_cube_lift3.mp4](demos_videos/bc_alone_blue_cube_lift3.mp4) | 第三个 lift case |
| [v10_sac_curriculum_yellow_placed.mp4](demos_videos/v10_sac_curriculum_yellow_placed.mp4) | SAC 课程化下完整 pick-and-place |
| [v10_sac_zero_shot_full_placed.mp4](demos_videos/v10_sac_zero_shot_full_placed.mp4) | SAC zero-shot 全场景偶发成功 |

详细迭代过程：[实验日志.md](实验日志.md)（含 v1-v11 SAC + v1-v9 expert + BC + SACfD 全部攻防战）。

## 目录结构

```
05_强化学习抓取/
├── env/                          # gymnasium 环境
│   ├── grasp_env.py              # GraspEnv（28-dim obs / 6-dim joint-space action）
│   ├── scene_constants.py        # 场景常量、关节、夹爪、阈值
│   ├── randomization.py          # 物体随机化（含课程化 target_override）
│   └── reward_shaping.py         # phase-gated dense reward
├── configs/
│   └── sac_default.yaml          # SAC 超参
├── train_sac.py                  # 训练（支持 --load-from 继续训练 + buffer 持久化）
├── eval.py                       # 单 ckpt 评估 + 视频
├── eval_final.py                 # 综合评估（训练分布 + zero-shot）
├── diagnose.py                   # 详细行为统计（approached/lifted/held/placed/oob）
├── expert.py                     # IK + scripted 抓取（baseline）
├── smoke_test.py                 # 50 步随机动作 sanity check
├── 实验日志.md                    # v1-v11 完整迭代记录
└── README.md                     # 本文件
```

## 快速复现

### 安装
```bash
pip install gymnasium stable-baselines3 tensorboard
# mujoco / numpy / opencv-python 应该已经从 04 阶段装好
```

### Smoke test
```bash
python smoke_test.py
```

### 训练（从零）
```bash
# Stage 1: yellow zone, radius 0.05, 600k step（~10 min）
python train_sac.py --total-timesteps 600000 --n-envs 8 \
    --obj-radius 0.05 --target yellow_cylinder \
    --run-name m1_yellow_v10_600k

# Stage 2: 扩展到 radius 0.10, load stage 1（~3-4 min）
python train_sac.py --total-timesteps 200000 --n-envs 8 \
    --obj-radius 0.10 --target yellow_cylinder \
    --load-from checkpoints/m1_yellow_v10_600k/best_model.zip \
    --run-name m1_yellow_s2_v3_buf
```

### 评估
```bash
# 综合评估（训练分布 + zero-shot, 各 100 ep + 视频）
python eval_final.py

# 详细行为诊断
python diagnose.py --ckpt checkpoints/m1_yellow_s2_v3_buf/best_model.zip \
    --target yellow_cylinder --obj-radius 0.10 --episodes 50

# 单 ckpt 评估 + 录视频
python eval.py --ckpt checkpoints/m1_yellow_s2_v3_buf/best_model.zip \
    --target yellow_cylinder --episodes 30 --render
```

### TensorBoard
```bash
tensorboard --logdir logs/
```

## 设计要点

### Action Space — 6-dim 关节增量
```python
[Δq1, Δq2, Δq3, Δq4, Δq5, gripper]    each in [-1, 1]
```
- Δq*: ±JOINT_DELTA_MAX (0.05 rad) 关节增量
- gripper: 线性映射到 [GRIPPER_CLOSE=-45°, GRIPPER_OPEN=90°]

为什么不用 EE-space + IK：单次 DLS IK 多步累积漂移，多重启 IK 太慢。关节空间动作直接稳定，SO-100 仅 5 DOF 探索空间也能接受。

### Observation Space — 28-dim 状态向量
```
arm_qpos(5) + arm_qvel(5) + gripper_qpos(1)
+ ee_pos(3) + obj_pos(3) + target_pos(3)
+ ee_to_obj(3) + obj_to_target(3)
+ contact_flag(1) + t_norm(1)
```

### Reward — Phase-gated Dense
Phase A（未抓起）：reach + lift_linear，引导接近 + 抬起。
Phase B（held=True）：held_baseline + transport（势能项），稳定运送到 zone。
Milestones：first_lift_bonus +20（一次性）、placed_bonus +200（终止）。
Penalties：step -0.01、action -0.001‖a‖²、OOB -10。

完整配方见 [env/reward_shaping.py](env/reward_shaping.py)，调试历程见 [实验日志.md](实验日志.md)。

### Reverse Curriculum
物体不直接全场景随机化，先在 zone 附近 ±5cm 出现，agent 学会"短链 grasp+drop"，再渐进扩大到 ±10cm。每 stage radius 扩展不能 > 2 倍，否则会大幅退步（实测 0.10 → 0.20 退步 -34%）。

## 已踩过的关键坑

1. **MuJoCo + multiprocessing fork 段错误** → `SubprocVecEnv(start_method="spawn")`
2. **VecEnv callback freq 是 vec_steps** → `freq // n_envs`
3. **arm 撞地板下不去** → 三组 collision groups（floor/obj/arm 分开）
4. **`SAC.load()` 不加载 replay buffer** → 显式 save/load_replay_buffer，否则 fine-tune 必崩
5. **课程化 zone_red (边缘) 比全场景还难** → 选中心 zone (zone_yellow)
6. **训练后期 critic 漂移** → 用 best_model.zip，不用 final.zip + LR linear decay
7. **7 类 reward gaming hack** → 详见 [实验日志.md](实验日志.md)

## 当前限制 / 下一步

- **placed 47%（训练分布）/ 9%（zero-shot 全场景）** —— 纯 SAC 路线极限
- **BC alone lift 80%（最难场景）** —— 但学不到 placed（局部 imitation）
- **SACfD（BC + SAC fine-tune）失败** —— critic 随机初始化，几千步内洗掉 BC

### 已验证可行 / 不可行

| 方案 | 状态 |
|------|------|
| 反向课程 SAC | ✓ placed 47%（受限场景） |
| Expert 几何调试 | ✓ placed 20% (30 ep, blue_cube) |
| BC pretrain | ✓ lift 80%, placed 0%（局限） |
| SACfD naive | ✗ actor 被随机 critic 洗 |
| SACfD with lr/10 | ✗ 同样被洗 |

### 未实施的下一步

- **Critic-only warmup**：用 demo offline 训 critic 数千步再放 actor（最有希望）
- **AWAC / IQL**：BC + RL 的 SOTA，actor loss 加 BC regularization
- **Multi-object expert**：调 yellow/red/green 的 GRIP_CLOSE_ACTION 收集多物体 demo

## BC 路线复现

```bash
# 1. 收集 50 个 placed demo（4 worker 并行，~3 hours）
python collect_demos.py --target blue_cube --n-success 50 --n-workers 4

# 2. BC pretrain (50 epochs, ~30 sec)
python bc_pretrain.py --demos demos/blue_cube_v9.npz \
    --target blue_cube --epochs 50 --out checkpoints/bc_blue_cube

# 3. eval BC alone（lift 80% / placed 0%）
python eval.py --ckpt checkpoints/bc_blue_cube/policy.zip \
    --target blue_cube --episodes 30 --render

# 4. (可选) SACfD fine-tune —— 当前会洗掉 BC，待 critic warmup 修复
python train_sac.py --target blue_cube --total-timesteps 50000 --n-envs 8 \
    --load-from checkpoints/bc_blue_cube/policy.zip \
    --load-demos demos/blue_cube_v9.npz \
    --lr 3e-5 --run-name m1_sacfd
```
