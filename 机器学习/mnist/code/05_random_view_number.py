#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_random_view_number.py

从 CSV 随机抽取一行（20x20=400个像素值），把图片可视化。
- 默认 CSV: mnist_20x20_images.csv（每行400个0~255的灰度值）
- 可选标签 CSV: mnist_20x20_labels.csv（同索引的一行作为标签）
- 支持设置放大倍数(scale)以便查看；可选择是否保存图片
"""

import os
import sys
import csv
import random
import argparse
import numpy as np
import cv2


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def reservoir_pick_line(file_path: str, skip_header: bool = False):
    """水塘抽样：单次遍历随机抽取一行，返回(索引, 行字符串)。"""
    chosen = None
    chosen_idx = -1
    with open(file_path, 'r', encoding='utf-8') as f:
        if skip_header:
            next(f, None)
        idx = -1
        for line in f:
            s = line.strip()
            if not s:
                continue
            idx += 1
            if random.random() < 1.0 / (idx + 1):
                chosen = s
                chosen_idx = idx
    if chosen is None:
        raise RuntimeError('CSV 文件为空或没有有效行')
    return chosen_idx, chosen


def parse_pixels(line: str) -> np.ndarray:
    """把逗号分隔的400个值解析为 20x20 uint8 灰度矩阵。"""
    parts = [p.strip() for p in line.split(',') if p.strip() != '']
    if len(parts) < 400:
        raise ValueError(f'该行只有 {len(parts)} 个值，期望 400')
    # 只取前400个，防止多余列
    vals = [int(float(x)) for x in parts[:400]]
    # 限制到 0..255
    vals = [0 if v < 0 else 255 if v > 255 else v for v in vals]
    arr = np.array(vals, dtype=np.uint8).reshape((20, 20))
    return arr


def load_label_by_index(label_csv: str, index: int, skip_header: bool = False):
    if not label_csv:
        return None
    with open(label_csv, 'r', encoding='utf-8') as f:
        if skip_header:
            next(f, None)
        for i, line in enumerate(f):
            if i == index:
                s = line.strip()
                if not s:
                    return None
                tokens = [t.strip() for t in s.split(',') if t.strip() != '']
                try:
                    return int(float(tokens[0])) if tokens else None
                except Exception:
                    return tokens[0] if tokens else None
    return None


def visualize(gray20: np.ndarray, scale: int = 10, save_path: str | None = None, title: str = 'Random 20x20'):
    # 放大显示以便观察像素格（使用最近邻避免模糊）
    if scale and scale > 1:
        img = cv2.resize(gray20, (20 * scale, 20 * scale), interpolation=cv2.INTER_NEAREST)
    else:
        img = gray20

    if save_path:
        ensure_dir_for_file(save_path)
        cv2.imwrite(save_path, img)

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description='从CSV随机抽取一张20x20灰度图并显示')
    parser.add_argument('--csv', type=str, default='mnist_20x20_images.csv', help='图片CSV路径')
    parser.add_argument('--labels', type=str, default='mnist_20x20_labels.csv', help='可选：标签CSV路径，同索引读取')
    parser.add_argument('--skip-header', action='store_true', help='若CSV第一行是表头则勾选')
    parser.add_argument('--scale', type=int, default=10, help='显示放大倍数(默认10 -> 200x200)')
    parser.add_argument('--save', type=str, default=None, help='可选：保存放大的20x20图片到此路径')
    parser.add_argument('--seed', type=int, default=None, help='随机种子(可复现实验)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not os.path.exists(args.csv):
        print(f'CSV 不存在: {args.csv}')
        sys.exit(1)

    try:
        idx, line = reservoir_pick_line(args.csv, skip_header=args.skip_header)
        gray20 = parse_pixels(line)
    except Exception as e:
        print(f'解析CSV失败: {e}')
        sys.exit(1)

    label = None
    if args.labels and os.path.exists(args.labels):
        try:
            label = load_label_by_index(args.labels, idx, skip_header=args.skip_header)
        except Exception:
            label = None

    title = f'Index={idx}' + (f' Label={label}' if label is not None else '')
    print(f"随机抽取：第 {idx+1} 行")
    visualize(gray20, scale=args.scale, save_path=args.save, title=title)


if __name__ == '__main__':
    main()