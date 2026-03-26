import numpy as np
from pathlib import Path
import joblib
import argparse


def load_txt_20x20(txt_path: Path) -> np.ndarray:
    """读取20x20的灰度文本，返回形状(20,20)的数组。"""
    arr = np.loadtxt(txt_path, dtype=np.float32)
    if arr.ndim == 1:
        # 如果是一行包含400个数
        if arr.size == 400:
            arr = arr.reshape(20, 20)
        else:
            raise ValueError(f"期望20x20=400个值，实际为{arr.size}")
    elif arr.shape != (20, 20):
        raise ValueError(f"期望尺寸20x20，实际为{arr.shape}")
    return arr


def predict_from_txt(model_path: Path, txt_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    if not txt_path.exists():
        raise FileNotFoundError(f"未找到输入文本: {txt_path}")

    # 读取并展开为(1,400)，归一化到[0,1]
    grid = load_txt_20x20(txt_path)
    X = grid.reshape(1, -1) / 255.0

    # 加载模型并推理
    model = joblib.load(model_path)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]  # (10,)

    # 输出结果
    print(f"预测数字: {pred}")
    print("概率分布(类别: 概率):")
    prob_map = {cls: float(proba[cls]) for cls in range(proba.shape[0])}
    print(prob_map)
    top3 = np.argsort(proba)[::-1][:3]
    print("Top-3:", [(int(c), float(proba[c])) for c in top3])


def main():
    parser = argparse.ArgumentParser(description="使用mnist_20x20_mlp.joblib对20x20灰度文本进行推理")
    parser.add_argument("--model", default="mnist_20x20_mlp.joblib", help="模型文件路径")
    parser.add_argument("--txt", default="gray_20x20.txt", help="20x20灰度文本路径")
    args = parser.parse_args()

    here = Path(__file__).parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = here / model_path
    txt_path = Path(args.txt)
    if not txt_path.is_absolute():
        txt_path = here / txt_path

    predict_from_txt(model_path, txt_path)


if __name__ == "__main__":
    main()