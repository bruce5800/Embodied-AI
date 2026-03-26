"""
MuJoCo 交互式仿真 Demo - 智能分拣场景
======================================
机械臂结构：
  joint1~5: 控制臂体姿态（底座/肩/肘/腕俯仰/腕旋转）
  joint6:   夹爪开合（FF.stl=固定半侧，GG.stl=活动半侧，旋转实现夹合）

运行：
  python mujoco_demo.py              # 交互式查看器
  python mujoco_demo.py --mode cv    # OpenCV 离屏渲染

键位：
  1~5       选择臂体关节
  ← →       控制当前关节
  ↑ ↓       调整步长 (1°~20°)
  Space     夹爪 开/合
  7/8/9     预设姿态
  0         复位
"""

import argparse
import os
import time
import numpy as np

# ─── 路径 ───
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_SCRIPT_DIR, "asserts", "lift_cube2.xml")

# ─── 执行器布局（共 6 个：joint1~5 控制臂，joint6 控制夹爪）───
ARM_ACTUATORS = 5        # joint1~5 的执行器索引 0~4
GRIPPER_IDX = 5          # joint6 的执行器索引 = 5
GRIPPER_OPEN = np.radians(90)    # 张开角度 90°
GRIPPER_CLOSE = np.radians(-5)  # 闭合角度 -5°

# ─── GLFW 方向键码 ───
GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT  = 263
GLFW_KEY_DOWN  = 264
GLFW_KEY_UP    = 265

# ─── 关节名称 ───
JOINT_NAMES = ['底座', '肩部', '肘部', '腕俯仰', '腕旋转']

# ─── 预设姿态（弧度），joint1~5 ───
PRESETS = {
    ord('7'): {
        'name': '就绪悬停',
        'joints': [0.0, -0.8, 0.5, 0.3, 0.0],
        'gripper': 'open',
    },
    ord('8'): {
        'name': '对准黄色小圆柱',
        'joints': [-0.58, -0.39, -1.4, -1.52, 0.0],
        'gripper': 'open',
    },
    ord('9'): {
        'name': '下探抓取',
        'joints': [-0.58, -0.39, -1.0, -1.52, 0.0],
        'gripper': 'open',
    },
}


def _clamp_joint(model, joint_targets, actuator_idx):
    """限幅到关节合法范围"""
    for ji in range(model.njnt):
        jnt = model.joint(ji)
        if jnt.type[0] in (2, 3):  # slide or hinge
            try:
                ai = model.actuator(f"{jnt.name}_actuator").id
                if ai == actuator_idx:
                    lo, hi = model.jnt_range[ji]
                    joint_targets[actuator_idx] = np.clip(
                        joint_targets[actuator_idx], lo, hi)
                    return
            except KeyError:
                continue


def _apply_preset(preset, joint_targets, gripper_state):
    """应用预设姿态"""
    for i, val in enumerate(preset['joints']):
        joint_targets[i] = val
    if preset['gripper'] == 'open':
        joint_targets[GRIPPER_IDX] = GRIPPER_OPEN
        gripper_state[0] = True
    else:
        joint_targets[GRIPPER_IDX] = GRIPPER_CLOSE
        gripper_state[0] = False
    print(f"  预设: {preset['name']}")


def _print_status(joint_targets, selected_joint, joint_step, gripper_state):
    """打印状态"""
    angles = [np.degrees(joint_targets[i]) for i in range(ARM_ACTUATORS)]
    grip = "开" if gripper_state[0] else "合"
    grip_deg = np.degrees(joint_targets[GRIPPER_IDX])
    parts = [f"J{i+1}:{a:+.0f}°" for i, a in enumerate(angles)]
    print(f"  [{JOINT_NAMES[selected_joint]}] 步长:{np.degrees(joint_step):.0f}°  "
          f"夹爪:{grip}({grip_deg:+.0f}°)  {' '.join(parts)}")


def run_interactive():
    """交互式查看器"""
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # 打印模型信息
    print("=" * 60)
    print(f"关节数: {model.njnt}  执行器数: {model.nu}  物体数: {model.nbody}")
    print("-" * 60)
    for i in range(model.njnt):
        jnt = model.joint(i)
        jtype = ["free", "ball", "slide", "hinge"][jnt.type[0]]
        if jtype in ("hinge", "slide"):
            lo, hi = model.jnt_range[i]
            if jtype == "hinge":
                lo, hi = np.degrees(lo), np.degrees(hi)
            print(f"  [{i}] {jnt.name:20s} ({jtype})  [{lo:+.0f}, {hi:+.0f}]")
        else:
            print(f"  [{i}] {jnt.name:20s} ({jtype})")
    print("=" * 60)

    # 控制状态
    joint_targets = np.zeros(model.nu)
    selected_joint = 0
    joint_step = np.radians(5)
    gripper_state = [True]  # True=开

    # 初始夹爪张开
    joint_targets[GRIPPER_IDX] = GRIPPER_OPEN

    def key_callback(keycode):
        nonlocal selected_joint, joint_step

        # 1~5: 选择臂体关节
        if ord('1') <= keycode <= ord('5'):
            selected_joint = keycode - ord('1')
            print(f"  选中: 关节{selected_joint+1} - {JOINT_NAMES[selected_joint]}")
            return

        # ← →: 控制选中关节
        if keycode == GLFW_KEY_RIGHT:
            joint_targets[selected_joint] += joint_step
            _clamp_joint(model, joint_targets, selected_joint)
            _print_status(joint_targets, selected_joint, joint_step, gripper_state)
            return
        if keycode == GLFW_KEY_LEFT:
            joint_targets[selected_joint] -= joint_step
            _clamp_joint(model, joint_targets, selected_joint)
            _print_status(joint_targets, selected_joint, joint_step, gripper_state)
            return

        # ↑ ↓: 步长
        if keycode == GLFW_KEY_UP:
            joint_step = min(joint_step * 2, np.radians(20))
            print(f"  步长: {np.degrees(joint_step):.0f}°")
            return
        if keycode == GLFW_KEY_DOWN:
            joint_step = max(joint_step / 2, np.radians(1))
            print(f"  步长: {np.degrees(joint_step):.0f}°")
            return

        # Space: 夹爪开合
        if keycode == ord(' '):
            gripper_state[0] = not gripper_state[0]
            joint_targets[GRIPPER_IDX] = GRIPPER_OPEN if gripper_state[0] else GRIPPER_CLOSE
            state = "张开" if gripper_state[0] else "闭合"
            print(f"  夹爪: {state} ({np.degrees(joint_targets[GRIPPER_IDX]):+.0f}°)")
            return

        # 0: 复位
        if keycode == ord('0'):
            joint_targets[:ARM_ACTUATORS] = 0
            joint_targets[GRIPPER_IDX] = GRIPPER_OPEN
            gripper_state[0] = True
            mujoco.mj_resetData(model, data)
            print("  复位")
            return

        # 7/8/9: 预设
        if keycode in PRESETS:
            _apply_preset(PRESETS[keycode], joint_targets, gripper_state)
            return

    print()
    print("╔══════════════════════════════════════════╗")
    print("║           键 盘 操 控 说 明              ║")
    print("╠══════════════════════════════════════════╣")
    print("║  1~5       选择臂体关节                  ║")
    print("║  ← →       控制当前关节                  ║")
    print("║  ↑ ↓       调整步长 (1°~20°)             ║")
    print("║  Space     夹爪 开/合 (joint6 旋转)      ║")
    print("║  0         复位                          ║")
    print("║  7         预设: 就绪悬停                ║")
    print("║  8         预设: 对准黄色圆柱            ║")
    print("║  9         预设: 下探抓取                ║")
    print("╠══════════════════════════════════════════╣")
    print("║  鼠标: 左键旋转 | 右键平移 | 滚轮缩放   ║")
    print("╚══════════════════════════════════════════╝")
    print()

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            data.ctrl[:] = joint_targets
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def run_cv_mode():
    """离屏渲染 + OpenCV"""
    import mujoco
    import cv2

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)

    cameras = ["camera_front", "camera_top", "camera_side"]
    cam_idx = 0
    selected_joint = 0
    joint_step = np.radians(5)
    gripper_state = [True]

    joint_targets = np.zeros(model.nu)
    joint_targets[GRIPPER_IDX] = GRIPPER_OPEN

    print("OpenCV 模式 | 1~5:选关节 ←→:控制 ↑↓:步长 Space:夹爪 0:复位 9:切相机 ESC:退出")

    while True:
        data.ctrl[:] = joint_targets
        for _ in range(10):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cameras[cam_idx])
        img = renderer.render()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # HUD
        cv2.putText(img_bgr, f"Cam: {cameras[cam_idx]}  Joint: {selected_joint+1}-{JOINT_NAMES[selected_joint]}  Step: {np.degrees(joint_step):.0f} deg",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        for i in range(ARM_ACTUATORS):
            a = np.degrees(joint_targets[i])
            color = (0, 255, 255) if i == selected_joint else (200, 200, 200)
            cv2.putText(img_bgr, f"J{i+1} {JOINT_NAMES[i]:4s}: {a:+6.1f}",
                        (10, 50 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        grip_str = "OPEN" if gripper_state[0] else "CLOSED"
        grip_color = (0, 255, 0) if gripper_state[0] else (0, 0, 255)
        grip_deg = np.degrees(joint_targets[GRIPPER_IDX])
        cv2.putText(img_bgr, f"Gripper: {grip_str} ({grip_deg:+.0f} deg)",
                    (10, 50 + ARM_ACTUATORS * 20 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, grip_color, 1)

        cv2.imshow("MuJoCo Sim", img_bgr)

        key = cv2.waitKey(16) & 0xFFFF
        if key == 27:
            break
        elif ord('1') <= key <= ord('5'):
            selected_joint = key - ord('1')
        elif key in (83, 3, 65363):     # →
            joint_targets[selected_joint] += joint_step
            _clamp_joint(model, joint_targets, selected_joint)
        elif key in (81, 2, 65361):     # ←
            joint_targets[selected_joint] -= joint_step
            _clamp_joint(model, joint_targets, selected_joint)
        elif key in (82, 0, 65362):     # ↑
            joint_step = min(joint_step * 2, np.radians(20))
        elif key in (84, 1, 65364):     # ↓
            joint_step = max(joint_step / 2, np.radians(1))
        elif key == ord(' '):
            gripper_state[0] = not gripper_state[0]
            joint_targets[GRIPPER_IDX] = GRIPPER_OPEN if gripper_state[0] else GRIPPER_CLOSE
        elif key == ord('0'):
            joint_targets[:ARM_ACTUATORS] = 0
            joint_targets[GRIPPER_IDX] = GRIPPER_OPEN
            gripper_state[0] = True
            mujoco.mj_resetData(model, data)
        elif key == ord('9'):
            cam_idx = (cam_idx + 1) % len(cameras)
        elif key in (ord('7'), ord('8')):
            _apply_preset(PRESETS[key], joint_targets, gripper_state)

    cv2.destroyAllWindows()
    renderer.close()


if __name__ == "__main__":
    import mujoco

    parser = argparse.ArgumentParser(description="MuJoCo 机械臂仿真 Demo")
    parser.add_argument("--mode", choices=["viewer", "cv"], default="viewer")
    args = parser.parse_args()

    if args.mode == "viewer":
        run_interactive()
    else:
        run_cv_mode()
