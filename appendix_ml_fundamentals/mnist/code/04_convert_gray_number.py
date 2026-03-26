#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_convert_gray_number.py

将输入图片压缩为 20x20，转换为灰度，并把像素值(0~255)写入文本文件：
- 黑色为 0，白色为 255，灰色为 [0,255]
- 默认输入为 roi_clean.jpg，输出灰度图 gray_20x20.png 和文本 gray_20x20.txt
- 支持选择插值方式和是否反相
"""

import os
import sys
import argparse
import cv2
import numpy as np


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def resize_to_20x20_gray(img, method: str = "area"):
    """把图像转换为灰度并缩放到 20x20。"""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    inter = interp_map.get(method.lower(), cv2.INTER_AREA)

    resized = cv2.resize(gray, (20, 20), interpolation=inter)
    resized = np.clip(resized, 0, 255).astype(np.uint8)
    return resized


def write_matrix_text(gray20, txt_path: str) -> None:
    """把 20x20 灰度矩阵写到文本文件，每行 20 个值，用空格分隔。"""
    ensure_dir_for_file(txt_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        for row in gray20:
            line = " ".join(str(int(v)) for v in row)
            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="图片压缩为20x20并导出灰度像素值文本")
    parser.add_argument("--input", "-i", type=str, default="roi_processed.jpg", help="输入图片路径，默认 roi_clean.jpg")
    parser.add_argument("--out-image", type=str, default="gray_20x20.png", help="输出20x20灰度图片路径")
    parser.add_argument("--out-text", type=str, default="gray_20x20.txt", help="输出像素值文本路径")
    parser.add_argument("--method", type=str, default="area", choices=["nearest", "linear", "cubic", "area", "lanczos"], help="缩放插值方式")
    parser.add_argument("--invert", action="store_true", help="反相：把白色变黑、黑色变白")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"无法读取图片: {args.input}")
        sys.exit(1)

    gray20 = resize_to_20x20_gray(img, method=args.method)
    if args.invert:
        gray20 = 255 - gray20

    # 保存灰度图
    ensure_dir_for_file(args.out_image)
    ok_img = cv2.imwrite(args.out_image, gray20)
    if not ok_img:
        print("保存灰度图失败！")
        sys.exit(1)

    # 保存文本矩阵
    write_matrix_text(gray20, args.out_text)

    print(f"已保存 20x20 灰度图: {os.path.abspath(args.out_image)}")
    print(f"已导出像素值文本: {os.path.abspath(args.out_text)}")


if __name__ == "__main__":
    main()