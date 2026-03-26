"""
MuJoCo 交互式仿真 Demo
======================
功能：
  1. 交互式 3D 查看器（鼠标旋转/缩放/拖拽物体）
  2. 键盘控制 6 个关节
  3. 离屏渲染 → OpenCV 显示（模拟相机视角，后续接 YOLO）

安装依赖：
  pip install mujoco opencv-python numpy

运行：
  python mujoco_demo.py              # 默认: 交互式查看器
  python mujoco_demo.py --mode cv    # 离屏渲染 + OpenCV 显示
"""

import argparse
import time
import numpy as np

# ─── XML 路径 ───
SCENE_XML = "asserts/lift_cube2.xml"


def run_interactive():
    """方式一: 交互式查看器，可鼠标旋转/缩放，双击物体施加力"""
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # 打印关节信息
    print("=" * 50)
    print("关节信息:")
    for i in range(model.njnt):
        name = model.joint(i).name
        jnt_range = model.jnt_range[i]
        print(f"  [{i}] {name}: range=[{np.degrees(jnt_range[0]):.1f}°, {np.degrees(jnt_range[1]):.1f}°]")
    print("=" * 50)

    # ─── 键盘控制状态 ───
    joint_targets = np.zeros(model.nu)  # 目标关节角度（弧度）
    step_size = np.radians(2)           # 每次按键转 2°

    # 键盘映射: key → (joint_index, direction)
    key_map = {
        # 数字键增, 字母键减
        ord('1'): (0, +1), ord('Q'): (0, -1),  # joint1 - 底座旋转
        ord('2'): (1, +1), ord('W'): (1, -1),  # joint2 - 肩部
        ord('3'): (2, +1), ord('E'): (2, -1),  # joint3 - 肘部
        ord('4'): (3, +1), ord('R'): (3, -1),  # joint4 - 腕部俯仰
        ord('5'): (4, +1), ord('T'): (4, -1),  # joint5 - 腕部旋转
        ord('6'): (5, +1), ord('Y'): (5, -1),  # joint6 - 夹爪
    }

    def key_callback(keycode):
        if keycode in key_map:
            idx, direction = key_map[keycode]
            joint_targets[idx] += direction * step_size
            # 限幅
            jnt_range = model.jnt_range[idx + 1]  # +1 跳过 freejoint
            joint_targets[idx] = np.clip(
                joint_targets[idx],
                jnt_range[0],
                jnt_range[1],
            )

    print("\n键盘控制:")
    print("  1/Q → 底座旋转    2/W → 肩部")
    print("  3/E → 肘部        4/R → 腕部俯仰")
    print("  5/T → 腕部旋转    6/Y → 夹爪")
    print("  鼠标: 左键旋转 | 右键平移 | 滚轮缩放")
    print("  双击物体可施加力\n")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            # 设置控制信号
            data.ctrl[:] = joint_targets

            # 物理仿真步进
            mujoco.mj_step(model, data)

            # 同步渲染（~60fps）
            viewer.sync()
            time.sleep(1.0 / 60.0)


def run_cv_mode():
    """方式二: 离屏渲染 + OpenCV 显示，模拟真实相机"""
    import mujoco
    import cv2

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # 创建离屏渲染器
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)

    # 可用的相机视角（在 XML 中定义）
    cameras = ["camera_front", "camera_top", "camera_vizu"]
    cam_idx = 0

    print("OpenCV 渲染模式")
    print(f"  当前相机: {cameras[cam_idx]}")
    print("  按 C 切换相机 | 按 ESC 退出")
    print("  按 1-6 控制关节（+ 方向）| 按 Q-Y 控制关节（- 方向）")

    joint_targets = np.zeros(model.nu)
    step_size = np.radians(2)

    while True:
        # 仿真步进
        data.ctrl[:] = joint_targets
        mujoco.mj_step(model, data)

        # 渲染当前相机视角
        renderer.update_scene(data, camera=cameras[cam_idx])
        img = renderer.render()  # RGB numpy array (H, W, 3)

        # 转 BGR 给 OpenCV 显示
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 叠加 HUD 信息
        cv2.putText(
            img_bgr,
            f"Camera: {cameras[cam_idx]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # 显示关节角度
        for i in range(min(model.nu, 6)):
            angle_deg = np.degrees(data.qpos[i + 7])  # 跳过 freejoint 的 7 个自由度
            cv2.putText(
                img_bgr,
                f"J{i+1}: {angle_deg:+.1f} deg",
                (10, 60 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("MuJoCo Sim", img_bgr)

        # 键盘处理
        key = cv2.waitKey(16) & 0xFF  # ~60fps
        if key == 27:  # ESC
            break
        elif key == ord("c"):
            cam_idx = (cam_idx + 1) % len(cameras)
            print(f"  切换到相机: {cameras[cam_idx]}")
        elif key == ord("0"):
            # 复位
            joint_targets[:] = 0
            mujoco.mj_resetData(model, data)
            print("  复位")
        else:
            # 关节控制
            key_map = {
                ord("1"): (0, +1), ord("q"): (0, -1),
                ord("2"): (1, +1), ord("w"): (1, -1),
                ord("3"): (2, +1), ord("e"): (2, -1),
                ord("4"): (3, +1), ord("r"): (3, -1),
                ord("5"): (4, +1), ord("t"): (4, -1),
                ord("6"): (5, +1), ord("y"): (5, -1),
            }
            if key in key_map:
                idx, direction = key_map[key]
                joint_targets[idx] += direction * step_size

    cv2.destroyAllWindows()
    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo 机械臂仿真 Demo")
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
