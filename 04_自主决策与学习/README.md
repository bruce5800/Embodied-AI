# 04 自主决策与学习 — LLM 规划 + 脚本式抓取

基于 MuJoCo 的机械臂智能分拣系统：**YOLO 视觉感知 → 逆运动学求解 → 脚本式抓取 → LLM 自然语言任务规划**。本阶段产出的 `genkiarm/` 项目是 [05 强化学习抓取](../05_强化学习抓取) 的前置 —— GraspEnv 环境、IK 求解器、场景资产都直接复用本阶段。

## 目录结构

```
04_自主决策与学习/
└── genkiarm/                # fork 自 GenkiBot/lerobot，二次开发
    ├── 本阶段自己写的代码
    │   ├── grasp_pipeline.py    主入口（流程编排）
    │   ├── config.py            全局参数（场景、抓取/放置参数）
    │   ├── ik_solver.py         逆运动学（MuJoCo Jacobian + DLS）
    │   ├── motion_control.py    运动控制（5 阶段抓取 + 5 阶段放置）
    │   ├── task_planner.py      任务规划（zone/absolute/relative/line）
    │   ├── llm_planner.py       LLM 接入（DeepSeek 自然语言 → JSON）
    │   ├── vision_pipeline.py   视觉感知（YOLO 数据生成 + 实时检测）
    │   └── mujoco_demo.py       交互式仿真 demo
    │
    ├── asserts/                 MuJoCo 模型 + 场景 XML
    ├── dataset/                 YOLO 训练数据集（MuJoCo 自动生成）
    ├── yolo11n.pt               训练好的 YOLO 模型
    ├── runs/                    YOLO 训练输出
    │
    ├── 文档
    │   ├── README.md                GenkiBot 项目原文（fork 来源）
    │   ├── SORTING_DEMO.md          ★ 本阶段核心成果文档（架构 + 模块说明）
    │   └── lerobot_遥操作指南.md    上游 lerobot 库的遥操作命令参考
    │
    └── 上游 lerobot（Apache 2.0）
        ├── lerobot/                 Python 包源码
        ├── LICENSE / NOTICE.txt     上游 license
        ├── CONTRIBUTING.md          上游贡献指南
        ├── Makefile / pyproject.toml
        └── benchmarks/ media/
```

## 主要成果

详见 [genkiarm/SORTING_DEMO.md](genkiarm/SORTING_DEMO.md) —— 完整架构图、模块说明、参数调整、交互 demo 操作。

## 与 05 阶段的关系

- **本阶段**：脚本式抓取（IK + 5 阶段运动），适合静态目标，不需训练
- **05 阶段**：把 GraspEnv 包成 gymnasium env，用 SAC + 反向课程从零训 RL（placed 40%），并探索 BC/DAPG/Diffusion Policy。05 的 `env/grasp_env.py` 直接 reuse 04 的场景 XML（`asserts/lift_cube2.xml` 等）和 IK 求解逻辑

## 快速运行

```bash
cd genkiarm

# 默认分拣 — 所有物体各归对应区域
python grasp_pipeline.py

# LLM 自然语言指令
python grasp_pipeline.py --llm

# 交互式仿真 demo（键盘控制）
mjpython mujoco_demo.py
```

依赖安装与详细命令见 [genkiarm/README.md](genkiarm/README.md) 与 [genkiarm/SORTING_DEMO.md](genkiarm/SORTING_DEMO.md)。
