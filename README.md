# MuJoCo 智能分拣 Demo

基于 MuJoCo 物理仿真的机械臂智能分拣系统。从感知到决策到执行，完整展示 **YOLO 目标检测 → 逆运动学求解 → 抓取放置 → LLM 自然语言控制** 的全流程。

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│  用户指令层                                                    │
│  "把红色圆柱放到蓝色方块上面"  /  JSON指令  /  默认分拣规则         │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│  LLM 规划层 (llm_planner.py)                                  │
│  DeepSeek API: 自然语言 → 结构化 JSON 指令                      │
└──────────────┬───────────────────────────────────────────────┘
               │  [{"object": "red_cylinder", "mode": "relative",
               │    "ref_object": "blue_cube", "relation": "above"}]
┌──────────────▼───────────────────────────────────────────────┐
│  任务解析层 (task_planner.py)                                  │
│  zone / absolute / relative / line → 具体坐标 (x, y, z)       │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│  感知层 (vision_pipeline.py)                                  │
│  MuJoCo 渲染 → YOLO 推理 → 物体类别 + 3D 位置                   │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│  运动控制层 (ik_solver.py + motion_control.py)                │
│  Jacobian IK 求解 → 分阶段执行 → 抓取验证 → 失败重试              │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│  物理仿真层 (MuJoCo)                                           │
│  SO-100 机械臂 + 夹爪 + 4 个待分拣物体 + 4 个放置区               │
└──────────────────────────────────────────────────────────────┘
```

## 文件结构

```
genkiarm/
├── grasp_pipeline.py     # 主入口 — 流程编排
├── config.py             # 全局参数（场景、物体、抓取/放置参数）
├── ik_solver.py          # 逆运动学求解器（MuJoCo Jacobian + 阻尼最小二乘）
├── motion_control.py     # 运动控制（渲染、移动、抓取/放置动作）
├── task_planner.py       # 任务规划（放置指令解析）
├── llm_planner.py        # LLM 接入（DeepSeek 自然语言 → 指令）
├── vision_pipeline.py    # 视觉感知（数据生成 + YOLO 实时检测）
├── mujoco_demo.py        # 交互式仿真 Demo（键盘控制机械臂）
└── asserts/
    ├── lift_cube2.xml    # 分拣场景定义（物体、放置区、相机）
    ├── test4.xml         # SO-100 机械臂模型
    └── robots_meshes/    # 机械臂 STL 网格文件 (AA~GG.stl)
```

## 快速开始

### 1. 安装依赖

```bash
pip install mujoco numpy opencv-python ultralytics openai
```

### 2. 生成训练数据

从 MuJoCo 仿真自动生成 YOLO 训练数据集（随机化物体位置 + 多相机渲染 + 自动标注）：

```bash
python vision_pipeline.py generate --num 500
```

### 3. 训练 YOLO

```bash
yolo task=detect mode=train model=yolo11n.pt data=dataset/dataset.yaml epochs=80 imgsz=640
```

### 4. 运行分拣

```bash
# 默认分拣 — 所有物体各归对应区域
python grasp_pipeline.py

# 只分拣指定物体
python grasp_pipeline.py --target blue_cube

# LLM 交互模式 — 用自然语言下达指令
python grasp_pipeline.py --llm

# 加载 JSON 指令文件
python grasp_pipeline.py --instructions tasks.json

# 只抓取不放置（调试用）
python grasp_pipeline.py --no-place
```

## 核心模块说明

### 视觉感知 — vision_pipeline.py

两个功能合一：

- **generate** — 在 MuJoCo 中随机化物体位姿，通过 3 个相机（front / top / side）渲染图像，自动计算 BBox 标注，输出 YOLO 格式数据集
- **detect** — 加载训练好的 YOLO 模型，实时检测 MuJoCo 渲染画面中的物体

```bash
python vision_pipeline.py generate --num 500   # 生成数据
python vision_pipeline.py detect               # 实时检测
```

### 逆运动学 — ik_solver.py

使用 MuJoCo 内置的 `mj_jacSite` 计算末端 Jacobian，通过阻尼最小二乘（DLS）迭代求解：

```
Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
```

- 自适应阻尼：远离目标时 λ 大（大步快走），靠近时 λ 小（小步精调）
- 多次随机重启（8 次尝试），避免局部最小值
- 在临时 data 上求解，不影响仿真状态

### 运动控制 — motion_control.py

**抓取序列（5 阶段）：**

1. 就绪姿态 + 张开夹爪
2. 移动到预抓取位置（目标上方 8cm）
3. 下降到抓取位置
4. 闭合夹爪
5. 抬起（15cm）

**放置序列（5 阶段）：**

1. 移动到放置区上方悬停
2. 下降到放置高度
3. 松开夹爪
4. 抬起离开
5. 回到就绪位

抓取后自动验证（检查物体 z 坐标是否被抬起），失败则重新检测、重新抓取，最多重试 3 次。

### 任务规划 — task_planner.py

支持 4 种放置模式：

| 模式 | 用途 | 示例 |
|------|------|------|
| `zone` | 放到预定义区域 | `{"object": "red_cylinder", "mode": "zone", "zone": "zone_red"}` |
| `absolute` | 放到绝对坐标 | `{"object": "red_cylinder", "mode": "absolute", "position": [-0.3, 0.1, 0.0]}` |
| `relative` | 相对另一物体 | `{"object": "red_cylinder", "mode": "relative", "ref_object": "blue_cube", "relation": "above"}` |
| `line` | 多物体排成一排 | `{"objects": ["red_cylinder", "blue_cube"], "mode": "line", "start": [-0.4,-0.2,0], "end": [-0.4,0.2,0]}` |

### LLM 规划 — llm_planner.py

通过 DeepSeek API 将自然语言指令转换为上述结构化 JSON 指令：

```
用户: "把红色圆柱放到蓝色方块上面"
  ↓ DeepSeek
[{"object": "red_cylinder", "mode": "relative",
  "ref_object": "blue_cube", "relation": "above", "offset": 0.08}]
```

支持持续对话，每次执行后可继续下达新指令。可独立测试：

```bash
python llm_planner.py "把四个物体排成一排"
```

## 仿真场景

### 机械臂

SO-100（5 DOF + 1 DOF 夹爪）：

| 关节 | 类型 | 范围 | 功能 |
|------|------|------|------|
| joint1 | 旋转 (Z轴) | +/-100 deg | 底座旋转 |
| joint2 | 旋转 (Y轴) | +/-100 deg | 肩部俯仰 |
| joint3 | 旋转 (Y轴) | +/-100 deg | 肘部俯仰 |
| joint4 | 旋转 (Y轴) | +/-100 deg | 腕部俯仰 |
| joint5 | 旋转 (Z轴) | +/-180 deg | 腕部旋转 |
| joint6 | 旋转 (X轴) | -50 ~ 90 deg | 夹爪开合 |

### 场景物体

| 物体 | 尺寸 | 初始位置 |
|------|------|----------|
| 红色圆柱 | r=3cm h=9cm | (-0.20, -0.10) 近处 |
| 蓝色方块 | 7.5cm 边长 | (-0.35, +0.12) 中等 |
| 绿色球体 | r=3.75cm | (-0.50, -0.05) 远处 |
| 黄色圆柱 | r=2.5cm h=6cm | (-0.55, +0.15) 最远 |

### 放置区域

```
               zone_blue (-0.15, +0.35)
                    o

zone_green          zone_yellow                    [臂基座]
(-0.50, 0)          (-0.15, 0)                     (0, 0)
    o                    o

               zone_red (-0.15, -0.35)
                    o
```

## 参数调整

所有参数集中在 `config.py`，常用调整项：

```python
# 夹爪落点偏了 → 调 GRASP_OFFSET
GRASP_OFFSET = np.array([0.02, -0.02, 0.02])  # [dx, dy, dz]

# 夹不住 → 调闭合角度（更负 = 夹更紧）
GRIPPER_CLOSE = np.radians(-30)

# 抓取高度
PRE_GRASP_HEIGHT = 0.08    # 悬停高度
PLACE_HEIGHT = 0.08        # 放置松手高度

# IK 失败 → 增加重试次数
MAX_GRASP_ATTEMPTS = 3
```

## 交互式 Demo

除了自动分拣，还可以用键盘手动控制机械臂探索场景：

```bash
python mujoco_demo.py              # 3D 交互查看器
python mujoco_demo.py --mode cv    # OpenCV 离屏渲染
```

键位：`1~5` 选关节，`← →` 控制，`↑ ↓` 调步长，`Space` 夹爪开合，`7/8/9` 预设姿态。
