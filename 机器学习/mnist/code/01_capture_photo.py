#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 OpenCV 打开摄像头并拍一张照片。
- 默认使用设备索引 0（通常是内置或第一个摄像头）
- 支持可选的预览窗口（按空格/回车拍照，ESC 取消）
- 支持设置分辨率和预热时间
"""

import cv2
import time
import argparse
import os
import sys


def open_camera(device: int, width: int | None, height: int | None) -> cv2.VideoCapture | None:
    # 在 Windows 上使用 CAP_DSHOW，打开速度更快，兼容性更好
    if os.name == "nt":
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        return None

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def save_image(img, output_path: str) -> bool:
    try:
        ok = cv2.imwrite(output_path, img)
        return bool(ok)
    except Exception as e:
        print(f"保存失败: {e}")
        return False


def run(device: int = 0,
        output: str = "photo.jpg",
        width: int | None = None,
        height: int | None = None,
        warmup: float = 0.8,
        show: bool = False) -> None:
    cap = open_camera(device, width, height)
    if cap is None:
        print(f"无法打开摄像头（索引 {device}）。")
        sys.exit(1)

    # 预热：读取若干帧让曝光/白平衡稳定
    start = time.time()
    last_frame = None
    while time.time() - start < warmup:
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头帧失败。")
            break
        last_frame = frame
        if show:
            cv2.imshow("Camera", frame)
            # ESC 取消
            if cv2.waitKey(1) & 0xFF == 27:
                print("已取消。")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    # 获取照片
    captured = None
    if show:
        print("按空格/回车拍照，ESC取消。")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头帧失败。")
                break
            cv2.imshow("Camera", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (13, 32):  # Enter 或 Space
                captured = frame
                break
            elif k == 27:  # ESC
                print("已取消。")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
    else:
        captured = last_frame
        if captured is None:
            ret, frame = cap.read()
            captured = frame if ret else None

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if captured is None:
        print("未能获取照片。")
        sys.exit(1)

    # 确保输出目录存在
    output_dir = os.path.dirname(output) or "."
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"创建输出目录失败: {e}")

    ok = save_image(captured, output)
    if not ok:
        print("保存图片失败。")
        sys.exit(1)

    print(f"已保存照片到: {os.path.abspath(output)}")



def main() -> None:
    parser = argparse.ArgumentParser(description="使用 OpenCV 打开摄像头并拍照。")
    parser.add_argument("--device", type=int, default=0, help="摄像头索引（默认 0）。")
    parser.add_argument("--output", type=str, default="photo.jpg", help="输出文件路径。")
    parser.add_argument("--width", type=int, default=None, help="设置摄像头宽度。")
    parser.add_argument("--height", type=int, default=None, help="设置摄像头高度。")
    parser.add_argument("--warmup", type=float, default=0.8, help="预热秒数，默认 0.8。")
    parser.add_argument("--show", action="store_true", help="显示实时预览窗口。")
    args = parser.parse_args()

    run(device=args.device,
        output=args.output,
        width=args.width,
        height=args.height,
        warmup=args.warmup,
        show=args.show)


if __name__ == "__main__":
    main()