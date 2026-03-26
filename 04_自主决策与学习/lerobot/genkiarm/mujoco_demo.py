"""
MuJoCo 交互式仿真 Demo - 智能分拣场景
======================================
功能：
  1. 交互式 3D 查看器（鼠标旋转/缩放/拖拽物体）
  2. 键盘控制 6 个关节 + 夹爪开合
  3. 离屏渲染 → OpenCV 显示（模拟相机视角，后续接 YOLO）
  4. 预设姿态一键到位，方便快速测试抓取

安装依赖：
  pip install mujoco opencv-python numpy

运行：
  python mujoco_demo.py              # 默认: 交互式查看器
  python mujoco_demo.py --mode cv    # 离屏渲染 + OpenCV 显示

键位说明（viewer 模式，完全避开 MuJoCo 内置快捷键）：
  数字键 1~6 选择关节，← → 方向键控制方向
  ↑ ↓ 方向键调整步长

  Space → 夹爪开/合    0 → 复位
  7/8/9 → 预设姿态（快速到位测试抓取）
"""

import argparse
import os
import time
import numpy as np

# ─── XML 路径 ───
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_SCRIPT_DIR, "asserts", "lift_cube2.xml")

# ─── 执行器索引 ───
JOINT_ACTUATORS = 6
GRIPPER_LEFT_IDX = 6
GRIPPER_RIGHT_IDX = 7
GRIPPER_OPEN = 0.0       # 张开：两指间距 7cm
GRIPPER_CLOSE = 0.03     # 闭合：两指间距 1cm

# ─── GLFW 特殊键码（MuJoCo viewer 用 GLFW） ───
GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT  = 263
GLFW_KEY_DOWN  = 264
GLFW_KEY_UP    = 265

# ─── 预设姿态（弧度），方便一键到位测试 ───
PRESETS = {
    ord('7'): {
        'name': '就绪姿态（悬停在桌面上方）',
        'joints': [0.0, -0.8, 0.5, 0.3, 0.0, 0.0],
        'gripper': 'open',
    },
    ord('8'): {
        'name': '抓取预备（对准黄色小圆柱上方）',
        'joints': [-0.54, -0.39, -1.52, -1.56, 0.0, 0.0],
        'gripper': 'open',
    },
    ord('9'): {
        'name': '低位抓取（再下探一点）',
        'joints': [-0.54, -0.30, -1.52, -1.56, 0.0, 0.0],
        'gripper': 'open',
    },
}

# ─── 关节名称 ───
JOINT_NAMES = ['底座', '肩部', '肘部', '腕俯仰', '腕旋转', '末端']


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


def _apply_preset(preset, joint_targets, gripper_state):
    """应用预设姿态"""
    for i, val in enumerate(preset['joints']):
        joint_targets[i] = val
    if preset['gripper'] == 'open':
        joint_targets[GRIPPER_LEFT_IDX] = GRIPPER_OPEN
        joint_targets[GRIPPER_RIGHT_IDX] = GRIPPER_OPEN
        gripper_state[0] = True
    else:
        joint_targets[GRIPPER_LEFT_IDX] = GRIPPER_CLOSE
        joint_targets[GRIPPER_RIGHT_IDX] = GRIPPER_CLOSE
        gripper_state[0] = False
    print(f"  预设: {preset['name']}")


def _print_status(joint_targets, selected_joint, joint_step, gripper_state):
    """打印当前状态"""
    angles = [np.degrees(joint_targets[i]) for i in range(JOINT_ACTUATORS)]
    grip = "张开" if gripper_state[0] else "闭合"
    step_deg = np.degrees(joint_step)
    parts = [f"J{i+1}:{a:+.0f}°" for i, a in enumerate(angles)]
    print(f"  [当前关节 {selected_joint+1}-{JOINT_NAMES[selected_joint]}]  "
          f"步长:{step_deg:.0f}°  夹爪:{grip}  {' '.join(parts)}")


def run_interactive():
    """方式一: 交互式查看器 — 数字键选关节，方向键控制"""
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # 打印模型信息
    print("=" * 60)
    print(f"模型: {os.path.basename(SCENE_XML)}")
    print(f"关节数: {model.njnt}  执行器数: {model.nu}  物体数: {model.nbody}")
    print("-" * 60)
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
    print("=" * 60)

    # ─── 控制状态 ───
    joint_targets = np.zeros(model.nu)
    selected_joint = 0        # 当前选中的关节 (0~5)
    joint_step = np.radians(5)  # 默认步长 5°
    gripper_state = [True]

    # 初始夹爪张开
    joint_targets[GRIPPER_LEFT_IDX] = GRIPPER_OPEN
    joint_targets[GRIPPER_RIGHT_IDX] = GRIPPER_OPEN

    def key_callback(keycode):
        nonlocal selected_joint, joint_step

        # 数字键 1~6: 选择关节
        if ord('1') <= keycode <= ord('6'):
            selected_joint = keycode - ord('1')
            print(f"  选中关节 {selected_joint+1}: {JOINT_NAMES[selected_joint]}")
            return

        # ← → 方向键: 控制当前关节
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

        # ↑ ↓ 方向键: 调整步长
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
            pos = GRIPPER_OPEN if gripper_state[0] else GRIPPER_CLOSE
            joint_targets[GRIPPER_LEFT_IDX] = pos
            joint_targets[GRIPPER_RIGHT_IDX] = pos
            state = "张开" if gripper_state[0] else "闭合"
            print(f"  夹爪: {state}")
            return

        # 0: 复位
        if keycode == ord('0'):
            joint_targets[:JOINT_ACTUATORS] = 0
            joint_targets[GRIPPER_LEFT_IDX] = GRIPPER_OPEN
            joint_targets[GRIPPER_RIGHT_IDX] = GRIPPER_OPEN
            gripper_state[0] = True
            mujoco.mj_resetData(model, data)
            print("  复位")
            return

        # 7/8/9: 预设姿态
        if keycode in PRESETS:
            _apply_preset(PRESETS[keycode], joint_targets, gripper_state)
            return

    print()
    print("╔══════════════════════════════════════════╗")
    print("║           键 盘 操 控 说 明              ║")
    print("╠══════════════════════════════════════════╣")
    print("║  1~6       选择关节                      ║")
    print("║  ← →       控制当前关节 (-/+)            ║")
    print("║  ↑ ↓       调整步长 (1°~20°)             ║")
    print("║  Space     夹爪 开/合                    ║")
    print("║  0         复位                          ║")
    print("║  7         预设: 就绪悬停                ║")
    print("║  8         预设: 对准红色圆柱            ║")
    print("║  9         预设: 低位接近地面            ║")
    print("╠══════════════════════════════════════════╣")
    print("║  鼠标: 左键旋转 | 右键平移 | 滚轮缩放   ║")
    print("║  双击物体可施加力                        ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print(f"  当前选中: 关节1-{JOINT_NAMES[0]}  步长: {np.degrees(joint_step):.0f}°")
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
    """方式二: 离屏渲染 + OpenCV 显示"""
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
    joint_targets[GRIPPER_LEFT_IDX] = GRIPPER_OPEN
    joint_targets[GRIPPER_RIGHT_IDX] = GRIPPER_OPEN

    print("OpenCV 渲染模式")
    print("  1~6: 选关节  ← →: 控制  ↑ ↓: 步长")
    print("  Space: 夹爪  0: 复位  9: 切换相机  7/8: 预设  ESC: 退出")

    while True:
        data.ctrl[:] = joint_targets
        for _ in range(10):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cameras[cam_idx])
        img = renderer.render()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # HUD: 相机 + 选中关节
        cv2.putText(img_bgr, f"Camera: {cameras[cam_idx]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img_bgr, f"Joint {selected_joint+1}: {JOINT_NAMES[selected_joint]}  "
                    f"Step: {np.degrees(joint_step):.0f} deg",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # HUD: 关节角度（高亮选中关节）
        for i in range(JOINT_ACTUATORS):
            angle_deg = np.degrees(joint_targets[i])
            color = (0, 255, 255) if i == selected_joint else (200, 200, 200)
            marker = ">" if i == selected_joint else " "
            cv2.putText(img_bgr, f"{marker}J{i+1} {JOINT_NAMES[i]:4s}: {angle_deg:+6.1f} deg",
                        (10, 80 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # HUD: 夹爪状态
        grip_str = "OPEN" if gripper_state[0] else "CLOSED"
        grip_color = (0, 255, 0) if gripper_state[0] else (0, 0, 255)
        cv2.putText(img_bgr, f" Gripper: {grip_str}",
                    (10, 80 + JOINT_ACTUATORS * 22 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, grip_color, 2)

        cv2.imshow("MuJoCo Sim", img_bgr)

        key = cv2.waitKey(16) & 0xFFFF  # 用 0xFFFF 捕获特殊键
        if key == 27:
            break

        # 数字键选关节
        if ord('1') <= key <= ord('6'):
            selected_joint = key - ord('1')
        # 方向键（OpenCV 特殊键码，macOS/Linux 不同，用通用判断）
        elif key in (83, 3):    # → 右
            joint_targets[selected_joint] += joint_step
            _clamp_joint(model, joint_targets, selected_joint)
        elif key in (81, 2):    # ← 左
            joint_targets[selected_joint] -= joint_step
            _clamp_joint(model, joint_targets, selected_joint)
        elif key in (82, 0):    # ↑ 上
            joint_step = min(joint_step * 2, np.radians(20))
        elif key in (84, 1):    # ↓ 下
            joint_step = max(joint_step / 2, np.radians(1))
        elif key == ord(' '):
            gripper_state[0] = not gripper_state[0]
            pos = GRIPPER_OPEN if gripper_state[0] else GRIPPER_CLOSE
            joint_targets[GRIPPER_LEFT_IDX] = pos
            joint_targets[GRIPPER_RIGHT_IDX] = pos
        elif key == ord('0'):
            joint_targets[:JOINT_ACTUATORS] = 0
            joint_targets[GRIPPER_LEFT_IDX] = GRIPPER_OPEN
            joint_targets[GRIPPER_RIGHT_IDX] = GRIPPER_OPEN
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
