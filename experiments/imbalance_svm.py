"""
imbalance_svm.py — Thử nghiệm xử lý mất cân bằng lớp cho SVM

So sánh:
  1. Baseline: best config (không class_weight)
  2. Balanced: best config + class_weight='balanced'

Đánh giá trên test set.
"""

import io
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from sklearn.metrics import accuracy_score, f1_score

from algorithms.svm import SVMClassifier
from data.loader import load_dataset
from evaluation.evaluator import evaluate, print_result_full
from preprocessing.preprocessor import Preprocessor


DATA_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "uit-vsfc-sentiment"),
    os.path.join(PROJECT_ROOT, "data", "data", "uit-vsfc-sentiment"),
]


def find_data_dir():
    for data_dir in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(data_dir, "train.csv")):
            return data_dir
    return DATA_DIR_CANDIDATES[0]


def train_and_evaluate_svm(X_train, y_train, X_test, y_test, **kwargs):
    model = SVMClassifier(**kwargs)
    model.fit(X_train, y_train)
    result = evaluate(model, X_test, y_test)
    return result


def main():
    data_dir = find_data_dir()
    print(f"Load dataset from: {data_dir}")
    splits = load_dataset(data_dir)

    X_train, y_train = splits["train"]
    X_dev, y_dev = splits["dev"]
    X_test, y_test = splits["test"]

    # Merge train + dev
    X_train_dev = X_train + X_dev
    y_train_dev = list(y_train) + list(y_dev)

    print("\nPreprocess text...")
    preprocessor = Preprocessor()
    X_train_dev_clean = preprocessor.clean_batch(X_train_dev)
    X_test_clean = preprocessor.clean_batch(X_test)

    # Best config từ tune
    best_config = {
        "C": 2.0,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": False,
    }

    print("\n" + "=" * 70)
    print("  SVM CLASS IMBALANCE EXPERIMENT")
    print("=" * 70)

    # 1. Baseline
    print("\n[1] Baseline (no class_weight)")
    baseline_result = train_and_evaluate_svm(
        X_train_dev_clean, y_train_dev,
        X_test_clean, y_test,
        **best_config
    )
    print_result_full(baseline_result)

    # 2. Balanced
    print("\n[2] class_weight='balanced'")
    balanced_config = {**best_config, "class_weight": "balanced"}
    balanced_result = train_and_evaluate_svm(
        X_train_dev_clean, y_train_dev,
        X_test_clean, y_test,
        **balanced_config
    )
    print_result_full(balanced_result)

    # So sánh
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<20} {'Baseline':>12} {'Balanced':>12} {'Diff':>12}")
    print("-" * 70)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        b = baseline_result.get(metric, 0)
        bal = balanced_result.get(metric, 0)
        diff = bal - b
        print(f"  {metric:<20} {b*100:>11.2f}% {bal*100:>11.2f}% {diff*100:>+11.2f}%")
    print("=" * 70)

    # Neutral label
    baseline_neutral = next(
        (x for x in baseline_result["label_scores"] if x["label"] == 1), None
    )
    balanced_neutral = next(
        (x for x in balanced_result["label_scores"] if x["label"] == 1), None
    )
    if baseline_neutral and balanced_neutral:
        print("\n  Neutral label accuracy:")
        print(f"    Baseline : {baseline_neutral['percent']*100:.2f}%")
        print(f"    Balanced : {balanced_neutral['percent']*100:.2f}%")


if __name__ == "__main__":
    main()
