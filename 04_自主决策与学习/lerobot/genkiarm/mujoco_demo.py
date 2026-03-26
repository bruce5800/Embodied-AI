"""
MuJoCo 交互式仿真 Demo - 智能分拣场景
======================================
功能：
  1. 交互式 3D 查看器（鼠标旋转/缩放/拖拽物体）
  2. 键盘控制 6 个关节 + 夹爪开合
  3. 离屏渲染 → OpenCV 显示（模拟相机视角，后续接 YOLO）

安装依赖：
  pip install mujoco opencv-python numpy

运行：
  python mujoco_demo.py              # 默认: 交互式查看器
  python mujoco_demo.py --mode cv    # 离屏渲染 + OpenCV 显示

键位说明（viewer 模式）：
  关节控制用两排键，上排+、下排-，避开 MuJoCo 内置快捷键(QWER等)
  ┌───┬───┬───┬───┬───┬───┐
  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │  ← 正方向
  ├───┼───┼───┼───┼───┼───┤
  │ Z │ X │ C │ V │ B │ N │  ← 负方向
  └───┴───┴───┴───┴───┴───┘
  底座  肩   肘  腕俯 腕旋 末端

  Space → 夹爪开/合    0 → 复位
"""

import argparse
import os
import time
import numpy as np

# ─── XML 路径（基于脚本所在目录，确保从任意位置运行都能找到）───
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_SCRIPT_DIR, "asserts", "lift_cube2.xml")

# ─── 执行器索引 ───
JOINT_ACTUATORS = 6
GRIPPER_LEFT_IDX = 6
GRIPPER_RIGHT_IDX = 7
GRIPPER_OPEN = -0.02
GRIPPER_CLOSE = 0.005

# ─── 键位映射（避开 MuJoCo viewer 内置快捷键 QWER/Tab/Space 等）───
# 上排数字键: 正方向 (+)
# 下排 ZXCVBN: 负方向 (-)
KEY_MAP_VIEWER = {
    ord('1'): (0, +1), ord('Z'): (0, -1),   # joint1 - 底座旋转
    ord('2'): (1, +1), ord('X'): (1, -1),   # joint2 - 肩部
    ord('3'): (2, +1), ord('C'): (2, -1),   # joint3 - 肘部
    ord('4'): (3, +1), ord('V'): (3, -1),   # joint4 - 腕部俯仰
    ord('5'): (4, +1), ord('B'): (4, -1),   # joint5 - 腕部旋转
    ord('6'): (5, +1), ord('N'): (5, -1),   # joint6 - 末端旋转
}

# OpenCV 模式用小写
KEY_MAP_CV = {
    ord('1'): (0, +1), ord('z'): (0, -1),
    ord('2'): (1, +1), ord('x'): (1, -1),
    ord('3'): (2, +1), ord('c'): (2, -1),
    ord('4'): (3, +1), ord('v'): (3, -1),
    ord('5'): (4, +1), ord('b'): (4, -1),
    ord('6'): (5, +1), ord('n'): (5, -1),
}


def _clamp_joint(model, joint_targets, actuator_idx):
    """将执行器目标值限制在对应关节的合法范围内"""
    for ji in range(model.njnt):
        jnt = model.joint(ji)
        if jnt.type[0] == 3:  # hinge
            try:
                ai = model.actuator(f"{jnt.name}_actuator").id
                if ai == actuator_idx:
                    lo, hi = model.jnt_range[ji]
                    joint_targets[actuator_idx] = np.clip(
                        joint_targets[actuator_idx], lo, hi
                    )
                    return
            except KeyError:
                continue


def _reset(model, data, joint_targets, gripper_state):
    """复位机械臂和夹爪"""
    joint_targets[:JOINT_ACTUATORS] = 0
    joint_targets[GRIPPER_LEFT_IDX] = GRIPPER_OPEN
    joint_targets[GRIPPER_RIGHT_IDX] = GRIPPER_OPEN
    gripper_state[0] = True
    mujoco.mj_resetData(model, data)
    print("  复位")


def _toggle_gripper(joint_targets, gripper_state):
    """切换夹爪开合"""
    gripper_state[0] = not gripper_state[0]
    pos = GRIPPER_OPEN if gripper_state[0] else GRIPPER_CLOSE
    joint_targets[GRIPPER_LEFT_IDX] = pos
    joint_targets[GRIPPER_RIGHT_IDX] = pos
    state = "张开" if gripper_state[0] else "闭合"
    print(f"  夹爪: {state}")


def run_interactive():
    """方式一: 交互式查看器"""
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # 打印模型信息
    print("=" * 55)
    print(f"模型: {SCENE_XML}")
    print(f"关节数: {model.njnt}  执行器数: {model.nu}  物体数: {model.nbody}")
    print("-" * 55)
    print("关节信息:")
    for i in range(model.njnt):
        jnt = model.joint(i)
        name = jnt.name
        jtype = ["free", "ball", "slide", "hinge"][jnt.type[0]]
        if jtype in ("hinge", "slide"):
            lo, hi = model.jnt_range[i]
            unit = "°" if jtype == "hinge" else "m"
            if jtype == "hinge":
                lo, hi = np.degrees(lo), np.degrees(hi)
            print(f"  [{i}] {name:25s} ({jtype:5s})  range=[{lo:+.1f}, {hi:+.1f}]{unit}")
        else:
            print(f"  [{i}] {name:25s} ({jtype})")
    print("=" * 55)

    # ─── 控制状态 ───
    joint_targets = np.zeros(model.nu)
    joint_step = np.radians(2)
    gripper_state = [True]  # 用 list 包装，便于闭包修改

    # 初始夹爪张开
    joint_targets[GRIPPER_LEFT_IDX] = GRIPPER_OPEN
    joint_targets[GRIPPER_RIGHT_IDX] = GRIPPER_OPEN

    def key_callback(keycode):
        if keycode in KEY_MAP_VIEWER:
            idx, direction = KEY_MAP_VIEWER[keycode]
            joint_targets[idx] += direction * joint_step
            _clamp_joint(model, joint_targets, idx)

        elif keycode == ord(' '):
            _toggle_gripper(joint_targets, gripper_state)

        elif keycode == ord('0'):
            _reset(model, data, joint_targets, gripper_state)

    print("\n键盘控制:")
    print("  ┌───┬───┬───┬───┬───┬───┐")
    print("  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │  ← 正方向 (+)")
    print("  ├───┼───┼───┼───┼───┼───┤")
    print("  │ Z │ X │ C │ V │ B │ N │  ← 负方向 (-)")
    print("  └───┴───┴───┴───┴───┴───┘")
    print("  底座  肩   肘  腕俯 腕旋 末端")
    print()
    print("  Space → 夹爪开/合    0 → 复位")
    print("  鼠标: 左键旋转 | 右键平移 | 滚轮缩放")
    print("  双击物体可施加力\n")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            data.ctrl[:] = joint_targets
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def run_cv_mode():
    """方式二: 离屏渲染 + OpenCV 显示"""
    import mujoco
    import cv2

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)

    cameras = ["camera_front", "camera_top", "camera_side"]
    cam_idx = 0

    print("OpenCV 渲染模式")
    print(f"  当前相机: {cameras[cam_idx]}")
    print("  ┌───┬───┬───┬───┬───┬───┐")
    print("  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │  ← 正方向")
    print("  ├───┼───┼───┼───┼───┼───┤")
    print("  │ Z │ X │ C │ V │ B │ N │  ← 负方向")
    print("  └───┴───┴───┴───┴───┴───┘")
    print("  9 → 切换相机  |  Space → 夹爪  |  0 → 复位  |  ESC → 退出")

    joint_targets = np.zeros(model.nu)
    joint_step = np.radians(2)
    gripper_state = [True]

    joint_targets[GRIPPER_LEFT_IDX] = GRIPPER_OPEN
    joint_targets[GRIPPER_RIGHT_IDX] = GRIPPER_OPEN

    while True:
        # 仿真步进（多步保证物理稳定）
        data.ctrl[:] = joint_targets
        for _ in range(10):
            mujoco.mj_step(model, data)

        # 渲染
        renderer.update_scene(data, camera=cameras[cam_idx])
        img = renderer.render()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # HUD
        cv2.putText(img_bgr, f"Camera: {cameras[cam_idx]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for i in range(JOINT_ACTUATORS):
            angle_deg = np.degrees(joint_targets[i])
            cv2.putText(img_bgr, f"J{i+1}: {angle_deg:+6.1f} deg",
                        (10, 60 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)

        grip_str = "OPEN" if gripper_state[0] else "CLOSED"
        grip_color = (0, 255, 0) if gripper_state[0] else (0, 0, 255)
        cv2.putText(img_bgr, f"Gripper: {grip_str}",
                    (10, 60 + JOINT_ACTUATORS * 22 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, grip_color, 2)

        cv2.imshow("MuJoCo Sim", img_bgr)

        key = cv2.waitKey(16) & 0xFF
        if key == 27:
            break
        elif key == ord("9"):
            cam_idx = (cam_idx + 1) % len(cameras)
            print(f"  切换到相机: {cameras[cam_idx]}")
        elif key == ord(" "):
            _toggle_gripper(joint_targets, gripper_state)
        elif key == ord("0"):
            _reset(model, data, joint_targets, gripper_state)
        elif key in KEY_MAP_CV:
            idx, direction = KEY_MAP_CV[key]
            joint_targets[idx] += direction * joint_step

    cv2.destroyAllWindows()
    renderer.close()


if __name__ == "__main__":
    import mujoco  # 顶层导入，供 _reset 使用

    parser = argparse.ArgumentParser(description="MuJoCo 机械臂仿真 Demo - 智能分拣场景")
    parser.add_argument(
        "--mode",
        choices=["viewer", "cv"],
        default="viewer",
        help="viewer: 交互式3D查看器 | cv: OpenCV离屏渲染",
    )
    args = parser.parse_args()

    if args.mode == "viewer":
        run_interactive()
    else:
        run_cv_mode()
