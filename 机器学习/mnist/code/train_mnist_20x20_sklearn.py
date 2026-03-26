import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier


def load_data(images_path: Path, labels_path: Path):
    X = np.loadtxt(images_path, delimiter=",", dtype=np.float32)
    y = np.loadtxt(labels_path, dtype=np.int64)
    return X, y


def build_pipeline():
    # 标准化 + MLP 神经网络（输入400像素，输出10类概率）
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 128, 64),  # 两层隐藏层
            activation="relu",
            solver="adam",
            alpha=1e-4,            # L2 正则
            batch_size=128,
            learning_rate="adaptive",
            max_iter=500,
            early_stopping=True,   # 自动使用验证集早停
            random_state=42,
            verbose=False,
        )),
    ])


def main():
    here = Path(__file__).parent
    images_path = here / "mnist_20x20_images.csv"
    labels_path = here / "mnist_20x20_labels.csv"

    # 读取数据
    X, y = load_data(images_path, labels_path)
    # 像素归一化到 [0,1]
    X = X / 255.0

    # 划分训练/测试集（保持类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 构建并训练模型
    model = build_pipeline()
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 演示概率输出（前5个样本）
    y_pred5 = model.predict(X_test[:5])
    proba5 = model.predict_proba(X_test[:5])
    print("\nTop-3 probabilities for first 5 test samples:")
    for i in range(proba5.shape[0]):
        p = proba5[i]
        top3 = np.argsort(p)[::-1][:3]
        print(f"Sample {i}: pred={int(y_pred5[i])}, top3={( [(int(c), float(p[c])) for c in top3] )}")

    # 保存模型到文件
    out_path = here / "mnist_20x20_mlp.joblib"
    joblib.dump(model, out_path)
    print(f"\nModel saved to: {out_path}")


if __name__ == "__main__":
    main()