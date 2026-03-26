import argparse
from pathlib import Path
import numpy as np
import joblib


def main():
    parser = argparse.ArgumentParser(description="Load saved model and predict MNIST 20x20 CSV samples")
    parser.add_argument("--model", default="mnist_20x20_mlp.joblib", help="Path to saved joblib model")
    parser.add_argument("--input-csv", default="mnist_20x20_images.csv", help="Path to images CSV (rows=samples)")
    parser.add_argument("--labels-csv", default=None, help="Optional labels CSV to compare ground truth")
    parser.add_argument("--index", type=int, default=0, help="Predict a single sample at this index (0-based)")
    parser.add_argument("--batch", type=int, default=0, help="If >0, predict first N samples and print list")
    parser.add_argument("--proba", action="store_true", help="Print class probability distribution as well")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    X = np.loadtxt(args.input_csv, delimiter=",", dtype=np.float32)
    X = X / 255.0

    if args.batch and args.batch > 0:
        n = min(args.batch, X.shape[0])
        Xn = X[:n]
        preds = model.predict(Xn)
        print(f"Predictions[0:{n}]: {preds.tolist()}")
        if args.proba:
            proba = model.predict_proba(Xn)
            print("Probabilities (top-3 per sample):")
            for i in range(n):
                p = proba[i]
                top3 = np.argsort(p)[::-1][:3]
                print(f"  sample {i}: top3={( [(int(c), float(p[c])) for c in top3] )}")
        if args.labels_csv:
            y = np.loadtxt(args.labels_csv, dtype=np.int64)
            print(f"Ground truth[0:{n}]: {y[:n].tolist()}")
    else:
        idx = max(0, min(args.index, X.shape[0] - 1))
        sample = X[idx].reshape(1, -1)
        pred = int(model.predict(sample)[0])
        print(f"Index {idx} -> predicted: {pred}")
        if args.proba:
            p = model.predict_proba(sample)[0]
            top3 = np.argsort(p)[::-1][:3]
            print(f"Probabilities top3: {( [(int(c), float(p[c])) for c in top3] )}")
        if args.labels_csv:
            y = np.loadtxt(args.labels_csv, dtype=np.int64)
            print(f"True label: {int(y[idx])}")


if __name__ == "__main__":
    main()