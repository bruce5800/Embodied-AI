# 05 强化学习抓取 — SAC + Curriculum + BC

在 04 阶段（脚本式抓取 + LLM 任务规划）之上，探索两条 RL 路线：
1. **SAC + 反向课程学习**：从零训 RL，达成 placed 47%（受限场景）
2. **BC + Expert demo**：用脚本式 expert 收集 demo 训行为克隆，达成 lift 80%（最难场景）

## 当前最佳成果

### 真实可用成果：SAC 课程化（v10 stage 2）

`checkpoints/sac/m1_yellow_s2_v3_buf/best_model.zip` — 800k step 训练。

**严格诊断 50 ep（v10 训练时 env，不含 friction softening）**：

| 指标 | 数据 | 含义 |
|------|------|------|
| placed (in zone) | 40% | 物体在目标区（含假阳性） |
| **stably_held** (≥10 步连续 held) | **40%** | **真持续抓握** |
| **placed_via_held**（先 stably_held 再 placed） | **34%** | **真"抓-放"完整流程** |
| held_run_max 平均 | 29.5 步 | 平均握 30 步（持续抓取） |
| placed - placed_via_held = 6% | "推/捞 hack" | 没真抓但物体到 zone（少数偶发） |

> ⚠️ 之前 `eval_final.py` 报的 47% 是**老 env（不含 friction softening）下的数字**——但中间为了 expert/BC 调试加了 friction，env 不一致让 ckpt 表现下降。现在 env 加了 `soften_contacts=False` 默认参数恢复 v10 训练 env，**真实成果是 placed_via_held 34%**。

### 演示视频（demos_videos/）

| 视频 | 内容 |
|------|------|
| [v10_real_grasp_yellow_1_seed10001.mp4](demos_videos/v10_real_grasp_yellow_1_seed10001.mp4) | **真"抓-放"成功**：连续 ≥10 步抓握 + 放到 zone |
| [v10_real_grasp_yellow_2_seed10002.mp4](demos_videos/v10_real_grasp_yellow_2_seed10002.mp4) | 同上，第二例 |
| [v10_real_grasp_yellow_3_seed10006.mp4](demos_videos/v10_real_grasp_yellow_3_seed10006.mp4) | 同上，第三例 |
| [v10_push_hack_yellow_1_seed10015.mp4](demos_videos/v10_push_hack_yellow_1_seed10015.mp4) | **反例对比**：placed=True 但**没真抓握** —— 用机械臂"捞"物体到 zone |

### 探索失败的路线（但有经验积累）

| 路线 | 真实结果 | 失败原因 |
|------|---------|---------|
| BC alone (50 demos) | placed 0%, **真抓 0%** | unimodal Gaussian + 50 demos 学不到 piecewise sequential 行为；MSE/likelihood 都 mode collapse 到"gripper 永远开" |
| BC v4 (close ×3 weight) | placed 0%, tried_close 30% | actor 学到闭爪但 timing 错位 |
| BC + SACfD (naive / gentle) | actor 被洗 | 随机 critic 让 SAC update 把 BC 行为冲掉 |
| **Diffusion Policy** (164 demos, 150 epoch MPS) | placed 0%, **stably_held 12%, lifted 28%, contacted 94%** | DDIM `pred_a_0` 缺 clip → 推理发散到几百（已修）；修后能真抓 12%，但 chunk 全饱和 ±1 没法平滑 transport；本质仍是 BC + 164 demos 不够覆盖 transport 相位 |

> ⚠️ **关于 `info["lifted"]` 假信号**：之前 BC 评估的 lift_rate 80% **不是真抓取** —— `lifted` 仅判 `obj_z>6cm` 不区分"被夹住举起"vs"张开 gripper 推过 6cm 瞬间"。**真实抓取必须看 `held`（contact AND closing AND lifted 同时为真）**。BC 路线 `held=0%`。

### 演示视频（demos_videos/）

| 视频 | 内容 |
|------|------|
| [v10_sac_curriculum_yellow_placed.mp4](demos_videos/v10_sac_curriculum_yellow_placed.mp4) | **SAC 课程化下完整 pick-and-place**（项目唯一真实成功演示） |
| [v10_sac_zero_shot_full_placed.mp4](demos_videos/v10_sac_zero_shot_full_placed.mp4) | SAC zero-shot 全场景偶发成功（9% rate） |

## 目录结构

```
05_强化学习抓取/
├── env/                  GraspEnv 环境（所有方法共用）
├── diffusion_policy/     Diffusion Policy 库代码（self-contained）
├── methods/
│   ├── sac/              路线 A: SAC + 反向课程
│   ├── expert/           路线 B: IK 脚本式 expert + demo 收集
│   ├── dapg/             路线 C: BC + DAPG
│   └── diffusion/        路线 D: Diffusion Policy 入口
├── tools/                通用诊断 / 评估工具
├── checkpoints/          (按方法分: sac/, bc/, dapg/, diffusion/)
├── demos/                expert 收集的 npz
├── demos_videos/         成果展示视频
├── 实验日志.md            v1-v11 SAC + v1-v9 expert + BC + DAPG + Diffusion 全过程
└── PROJECT_STRUCTURE.md  详细结构与调用示例
```

详见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。

## 快速复现

### 安装
```bash
pip install gymnasium stable-baselines3 tensorboard
# mujoco / numpy / opencv-python 应该已经从 04 阶段装好
```

### Smoke test
```bash
python tools/smoke_test.py
```

### 训练（从零）
```bash
# Stage 1: yellow zone, radius 0.05, 600k step（~10 min）
python methods/sac/train.py --total-timesteps 600000 --n-envs 8 \
    --obj-radius 0.05 --target yellow_cylinder \
    --run-name m1_yellow_v10_600k

# Stage 2: 扩展到 radius 0.10, load stage 1（~3-4 min）
python methods/sac/train.py --total-timesteps 200000 --n-envs 8 \
    --obj-radius 0.10 --target yellow_cylinder \
    --load-from checkpoints/sac/m1_yellow_v10_600k/best_model.zip \
    --run-name m1_yellow_s2_v3_buf
```

### 评估
```bash
# 综合评估（训练分布 + zero-shot, 各 100 ep + 视频）
python tools/eval_final.py

# 详细行为诊断
python tools/diagnose.py --ckpt checkpoints/sac/m1_yellow_s2_v3_buf/best_model.zip \
    --target yellow_cylinder --obj-radius 0.10 --episodes 50

# 单 ckpt 评估 + 录视频
python tools/eval.py --ckpt checkpoints/sac/m1_yellow_s2_v3_buf/best_model.zip \
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

### 4 路线最终对比（严格 metrics: stably_held + placed_via_held）

| 路线 | placed | stably_held | 训练时长 | 真实评价 |
|------|--------|-------------|---------|---------|
| **A. SAC + 课程** (m1_yellow_s2_v3_buf) | **40%** | **40%** | 800k step / ~3h | ✓ **唯一能 placed 的方案** |
| B. Expert IK 脚本 (blue_cube_v10) | 20% | n/a | 0 (rule-based) | ✓ 用于收集 demo，非 learned policy |
| C. BC alone (50/164 demos) | 0% | 0% | 50 epoch / 30s | ✗ unimodal Gaussian mode collapse |
| C'. BC + SACfD | 0% | 0% | 50k step | ✗ 随机 critic 洗掉 BC |
| **D. Diffusion Policy** (164 demos) | **0%** | **12%** | 150 epoch / 30 min MPS | △ **真能抓但抓得不稳，无法 transport** |

> ⚠️ Diffusion Policy 期间踩了 2 个隐藏 bug：① MPS `non_blocking=True` → NaN（trainer.py 改同步），② DDIM scheduler 缺 `pred_a_0.clamp(-1,1)` → 推理发散到几百（scheduler.py 已修）。修完才有 12% 真抓，否则全 0%。

### 关于"为什么 Diffusion 比 SAC 还差"

- **SAC 直接在 env 里探索 + dense reward 引导** → 能学到完整 grasp+place 链路（800k step + 课程）
- **Diffusion/BC 只 imitate expert，不在 env 探索** → 能复制「approach + close」的局部模式，但 transport 相位需要细控制（chunk 内 Δq 平滑变化）；164 demos 不足以让模型学到 obs→精确 chunk 的条件映射，最终 chunk 全饱和 ±1 像 expert 但 phase timing 错乱
- **结论**：在我们这个 5DOF + hook gripper 任务上，**dense-reward RL > offline imitation**，跟 paper 上 7DOF 抓 + 数千 demo 的设置不一样

### 未实施的下一步

- **Critic-only warmup**：用 demo offline 训 critic 数千步再放 actor（最有希望）
- **AWAC / IQL**：BC + RL 的 SOTA，actor loss 加 BC regularization
- **更多 demo + 更大模型**：500+ demo + 5M+ params 让 Diffusion 真正发挥（需要 GPU）
- **Multi-object expert**：调 yellow/red/green 的 GRIP_CLOSE_ACTION 收集多物体 demo

## BC 路线复现

```bash
# 1. 收集 50 个 placed demo（4 worker 并行，~3 hours）
python methods/expert/collect_demos.py --target blue_cube --n-success 50 --n-workers 4

# 2. BC pretrain (50 epochs, ~30 sec)
python methods/dapg/bc_pretrain.py --demos demos/blue_cube_v9.npz \
    --target blue_cube --epochs 50 --out checkpoints/bc/bc_blue_cube_v4

# 3. eval BC alone（lift 80% / placed 0%）
python tools/eval.py --ckpt checkpoints/bc/bc_blue_cube_v4/policy.zip \
    --target blue_cube --episodes 30 --render

# 4. (可选) SACfD fine-tune —— 当前会洗掉 BC，待 critic warmup 修复
python methods/sac/train.py --target blue_cube --total-timesteps 50000 --n-envs 8 \
    --load-from checkpoints/bc/bc_blue_cube_v4/policy.zip \
    --load-demos demos/blue_cube_v9.npz \
    --lr 3e-5 --run-name m1_sacfd
```
