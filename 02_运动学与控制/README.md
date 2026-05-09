# 02 运动学与控制 — 学习阶段的零散探索

本阶段是机械臂正/逆运动学、主从控制、遥操作的学习与试验，每个子目录是一个独立的小实验。
**这些功能后续被整合进 [04 自主决策与学习](../04_自主决策与学习)** 形成完整 demo（IK 求解器、运动控制流水线），本目录保留作为单点学习记录。

## 子目录

| 子目录 | 内容 | 整合去向 |
|--------|------|---------|
| `forward_kinematics/` | 正运动学求解（DH 参数 / 几何法） | 04 `ik_solver.py` |
| `inverse_kinematics/` | 逆运动学求解（Jacobian + 阻尼最小二乘 / ikpy / 几何解析法） | 04 `ik_solver.py`、05 `env/grasp_env.py` |
| `master_slave_control/` | 主从遥操作（WebSocket 通信、teacher/student 角度同步、IK 多解法对比） | 04 `motion_control.py` |
| `teleoperation/` | 遥操作（HostProtocol 串口协议、playground 试验） | 部分整合到 04 motion_control |

## 命名说明

子目录里的脚本多以 `01_xxx.py / 02_xxx.py` 编号 —— 这是**学习顺序**，不是依赖顺序。每个文件可独立运行，作为某个知识点的最小验证。

## 关键试验

- `inverse_kinematics/inverse_kinematics.py` — Jacobian + DLS 阻尼最小二乘 IK，最终被 04/05 复用
- `master_slave_control/08_ik_demo_几何解析法.py` / `09_ik_demo_逼近法.py` — 几何法 vs 数值法对比
- `master_slave_control/01_websocket_server.py` + `websocket_test.html` — 浏览器 ↔ 机械臂通信
- `teleoperation/06_tele_operation.py` — 完整 teacher/student 同步遥操作

## 不需要看本目录的情况

如果只关心整合后的 demo 或最终成果，**直接看 [04 自主决策与学习](../04_自主决策与学习) 即可**。本目录是过程记录，对外部读者参考价值有限。
