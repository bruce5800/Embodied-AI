import os
import json
import argparse
from glob import glob

# 默认类别映射，可按需用 --classes 覆盖
DEFAULT_CLASS_MAP = {"erji": 0, "esp32": 1}


def parse_classes_arg(s: str):
    """支持两种格式：
    1) "erji=0,esp32=1" 显式编号
    2) "erji,esp32"    按顺序从0开始编号
    """
    m = {}
    s = s.strip()
    if "=" in s:
        for part in s.split(","):
            name, idx = part.split("=")
            m[name.strip()] = int(idx.strip())
    else:
        for i, name in enumerate([x.strip() for x in s.split(",") if x.strip()]):
            m[name] = i
    return m


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def convert_one(json_path: str, out_dir: str, class_map: dict):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    size = data.get("size", {})
    W = size.get("width")
    H = size.get("height")
    if not W or not H:
        raise ValueError(f"缺少图片尺寸信息: {json_path}")

    objects = data.get("outputs", {}).get("object", [])
    if isinstance(objects, dict):
        objects = [objects]

    lines = []
    for obj in objects:
        label = obj.get("name") or obj.get("label")
        box = obj.get("bndbox") or obj.get("bbox") or {}
        xmin, ymin = box.get("xmin"), box.get("ymin")
        xmax, ymax = box.get("xmax"), box.get("ymax")
        if None in (xmin, ymin, xmax, ymax) or label is None:
            # 跳过不完整标注
            continue

        # 计算归一化中心点与宽高
        x_center = ((xmin + xmax) / 2.0) / W
        y_center = ((ymin + ymax) / 2.0) / H
        w_norm = (xmax - xmin) / W
        h_norm = (ymax - ymin) / H

        # 夹取到[0,1]，避免越界
        def clamp(v):
            return max(0.0, min(1.0, float(v)))

        x_center = clamp(x_center)
        y_center = clamp(y_center)
        w_norm = clamp(w_norm)
        h_norm = clamp(h_norm)

        # 类别ID
        if label not in class_map:
            class_map[label] = max(class_map.values(), default=-1) + 1
        cid = class_map[label]

        lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # 写出同名 .txt 到输出目录
    base = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(out_dir, f"{base}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def write_classes_txt(out_dir: str, class_map: dict):
    # 以 id 从小到大写出映射，格式："id name"
    inv = sorted(((v, k) for k, v in class_map.items()), key=lambda x: x[0])
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for cid, name in inv:
            f.write(f"{cid} {name}\n")


def main():
    parser = argparse.ArgumentParser(description="将 JSON 标注转换为 YOLO txt 格式")
    parser.add_argument("--input-dir", default=r"e:\\sz_gypx02\\day05\\code\\yolo\\json", help="JSON 输入目录")
    parser.add_argument("--output-dir", default=None, help="txt 输出目录（默认同输入目录）")
    parser.add_argument("--classes", default=None, help="类别映射，例如 'erji=0,esp32=1' 或 'erji,esp32'")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    ensure_dir(output_dir)

    class_map = DEFAULT_CLASS_MAP.copy()
    if args.classes:
        class_map = parse_classes_arg(args.classes)

    json_files = sorted(glob(os.path.join(input_dir, "*.json")))
    if not json_files:
        print(f"未在 {input_dir} 找到 json 文件")
        return

    print(f"共发现 {len(json_files)} 个 JSON，开始转换…")
    out_files = []
    for jp in json_files:
        out_files.append(convert_one(jp, output_dir, class_map))
    write_classes_txt(output_dir, class_map)

    print(f"已写出 {len(out_files)} 个 txt 到: {output_dir}")
    print("示例：", out_files[:3])


if __name__ == "__main__":
    main()