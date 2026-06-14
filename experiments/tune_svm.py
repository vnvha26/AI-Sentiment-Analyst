"""
tune_svm.py — Hyperparameter tuning cho SVM trên dev set

Grid search các tham số:
  - C: [0.1, 0.5, 1.0, 2.0, 5.0]
  - ngram_range: [(1,1), (1,2), (1,3)]
  - max_features: [10000, 20000, 30000, 50000]
  - sublinear_tf: [True, False]

Kết quả in ra dạng bảng, chọn best theo dev_f1.
Lưu best config vào configs/svm_best.json (C7).
"""

import io
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from sklearn.metrics import accuracy_score, f1_score

from algorithms.svm import SVMClassifier
from data.loader import load_dataset
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


def main():
    data_dir = find_data_dir()
    print(f"Load dataset from: {data_dir}")
    splits = load_dataset(data_dir)

    X_train, y_train = splits["train"]
    X_dev, y_dev = splits["dev"]

    print("\nPreprocess text...")
    preprocessor = Preprocessor()
    X_train_clean = preprocessor.clean_batch(X_train)
    X_dev_clean = preprocessor.clean_batch(X_dev)

    param_grid = {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "ngram_range": [(1, 1), (1, 2), (1, 3)],
        "max_features": [10000, 20000, 30000, 50000],
        "sublinear_tf": [True, False],
    }

    print("\n" + "=" * 80)
    print("  SVM HYPERPARAMETER TUNING ON DEV SET")
    print("=" * 80)
    print(f"  Train: {len(X_train)} mẫu | Dev: {len(X_dev)} mẫu")
    print()

    results = []
    best_f1 = -1
    best_config = None

    total_configs = (
        len(param_grid["C"])
        * len(param_grid["ngram_range"])
        * len(param_grid["max_features"])
        * len(param_grid["sublinear_tf"])
    )
    config_idx = 0

    for C in param_grid["C"]:
        for ngram_range in param_grid["ngram_range"]:
            for max_features in param_grid["max_features"]:
                for sublinear_tf in param_grid["sublinear_tf"]:
                    config_idx += 1
                    print(
                        f"[{config_idx}/{total_configs}] "
                        f"C={C} ngram={ngram_range} max_feat={max_features} sublinear={sublinear_tf}"
                    )

                    t0 = time.time()
                    model = SVMClassifier(
                        C=C,
                        ngram_range=ngram_range,
                        max_features=max_features,
                        sublinear_tf=sublinear_tf,
                    )
                    model.fit(X_train_clean, y_train)
                    train_time = time.time() - t0

                    y_pred = model.predict(X_dev_clean)
                    dev_acc = accuracy_score(y_dev, y_pred)
                    dev_f1 = f1_score(y_dev, y_pred, average="weighted", zero_division=0)

                    print(f"  -> dev_acc={dev_acc*100:.2f}% dev_f1={dev_f1*100:.2f}% time={train_time:.1f}s")

                    config = {
                        "C": C,
                        "ngram_range": ngram_range,
                        "max_features": max_features,
                        "sublinear_tf": sublinear_tf,
                        "dev_accuracy": dev_acc,
                        "dev_f1": dev_f1,
                        "train_time": train_time,
                    }
                    results.append(config)

                    if dev_f1 > best_f1:
                        best_f1 = dev_f1
                        best_config = config

    print("\n" + "=" * 80)
    print("  TUNING RESULTS")
    print("=" * 80)
    print(
        f"  {'C':>4} {'ngram':>8} {'max_feat':>10} {'sublinear':>10} "
        f"{'dev_acc':>10} {'dev_f1':>10} {'time(s)':>10}"
    )
    print("-" * 80)
    for r in results:
        marker = " ⭐" if r is best_config else ""
        print(
            f"  {r['C']:>4.1f} {str(r['ngram_range']):>8} {r['max_features']:>10} "
            f"{str(r['sublinear_tf']):>10} {r['dev_accuracy']*100:>9.2f}% "
            f"{r['dev_f1']*100:>9.2f}% {r['train_time']:>9.1f}s{marker}"
        )
    print("=" * 80)

    if best_config:
        print(f"\n  🏆 BEST CONFIG (by dev_f1={best_f1*100:.2f}%):")
        for k, v in best_config.items():
            print(f"     {k}: {v}")

        # Lưu best config
        configs_dir = os.path.join(PROJECT_ROOT, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        best_path = os.path.join(configs_dir, "svm_best.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        print(f"\n  ✅ Saved best config to {best_path}")


if __name__ == "__main__":
    main()
