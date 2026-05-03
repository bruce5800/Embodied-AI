# 05 强化学习抓取 — SAC + Reverse Curriculum

在 04 阶段（脚本式抓取 + LLM 任务规划）之上，用 **SAC 强化学习 + 反向课程学习** 训练机械臂抓取-放置策略。

## 当前最佳

**v10 stage 2 best_model**（[checkpoints/m1_yellow_s2_v3_buf/best_model.zip](checkpoints/m1_yellow_s2_v3_buf/best_model.zip)）：

| 评估场景 | placed | held | lifted | OOB |
|---------|--------|------|--------|-----|
| 训练分布（zone_yellow ±10cm） | **47%** | 53% | 61% | 15% |
| zero-shot 全场景 | 9% | 9% | 23% | 19% |

100 episode 评估，single object = `yellow_cylinder`，target zone = `zone_yellow (-0.15, 0.0)`。

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

- **placed 47%（训练分布）/ 9%（zero-shot 全场景）** —— 离 production-ready 还有距离
- **stage 2.5+ (radius ≥ 0.20) 难学** —— SAC + reward shaping 在长 horizon transport 上撞墙
- **下一步候选**：
  - **B. HER (Hindsight Experience Replay)** —— sparse reward 长 horizon 经典解
  - **C. BC pretrain + SAC fine-tune** —— manipulation 工业标准做法（前提：先把 expert.py 调到 ≥ 50% placed）
