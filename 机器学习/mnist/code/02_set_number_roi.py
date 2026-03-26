#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_set_number_roi.py

在图片中用鼠标框选一个数字区域（ROI），并把该区域保存为新图片。
- 默认读取当前目录下的 `photo.jpg`
- 鼠标拖拽选择，按 回车/空格 确认，ESC 取消
- 支持限制显示窗口大小，避免大图溢出屏幕（坐标自动映射回原图）
"""

import cv2
import argparse
import os
import sys


def load_image(path: str):
    img = cv2.imread(path)
    return img


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def resize_for_display(img, max_w: int | None, max_h: int | None):
    """根据给定最大宽高把图片缩放用于显示，返回(显示图, 缩放比例)。
    比例=显示尺寸/原始尺寸；如果未缩放则为1.0。
    """
    if max_w is None and max_h is None:
        return img, 1.0

    h, w = img.shape[:2]
    scale = 1.0
    if max_w is not None and w > max_w:
        scale = min(scale, max_w / float(w))
    if max_h is not None and h > max_h:
        scale = min(scale, max_h / float(h))

    if scale < 1.0:
        disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = img
        scale = 1.0
    return disp, scale


def map_rect_to_original(rect, scale: float):
    """把在显示图上的矩形映射回原图坐标。"""
    x, y, w, h = rect
    if scale < 1.0:
        x = int(round(x / scale))
        y = int(round(y / scale))
        w = int(round(w / scale))
        h = int(round(h / scale))
    return x, y, w, h


def crop_roi(img, rect):
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return None
    H, W = img.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    x2 = max(0, min(x + w, W))
    y2 = max(0, min(y + h, H))
    return img[y:y2, x:x2]


def main() -> None:
    parser = argparse.ArgumentParser(description="用鼠标框选数字区域并保存ROI图片")
    parser.add_argument("--input", "-i", type=str, default="photo.jpg", help="输入图片路径，默认 photo.jpg")
    parser.add_argument("--output", "-o", type=str, default="roi.jpg", help="输出ROI图片路径，默认 roi.jpg")
    parser.add_argument("--max-width", type=int, default=1280, help="显示窗口最大宽度，默认 1280")
    parser.add_argument("--max-height", type=int, default=900, help="显示窗口最大高度，默认 900")
    args = parser.parse_args()

    img = load_image(args.input)
    if img is None:
        print(f"无法读取图片: {args.input}")
        sys.exit(1)

    disp, scale = resize_for_display(img, args.max_width, args.max_height)

    print("提示：用鼠标拖拽选择框，按 回车/空格 确认，按 ESC 取消。")
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    rect = cv2.selectROI("Select ROI", disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    if isinstance(rect, tuple) and len(rect) == 4:
        pass
    else:
        print("未选择ROI。")
        sys.exit(1)

    rect_on_original = map_rect_to_original(rect, scale)
    roi = crop_roi(img, rect_on_original)
    if roi is None or roi.size == 0:
        print("选择的ROI为空或越界。")
        sys.exit(1)

    ensure_dir_for_file(args.output)
    ok = cv2.imwrite(args.output, roi)
    if not ok:
        print("保存ROI失败！")
        sys.exit(1)

    print(f"ROI已保存到: {os.path.abspath(args.output)}")

    # 展示保存的ROI片段，任意键关闭
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()