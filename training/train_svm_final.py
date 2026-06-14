"""
train_svm_final.py — Train SVM final với best params trên train+dev

1. Đọc best config từ configs/svm_best.json
2. Merge train + dev
3. Train SVM với best params
4. Evaluate trên test
5. Lưu model vào models/svm_best.joblib
"""

import io
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from algorithms.svm import SVMClassifier
from data.loader import load_dataset
from evaluation.evaluator import evaluate, print_result_full
from preprocessing.preprocessor import Preprocessor
from utils.model_manager import save_model


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
    # 1. Load best config
    config_path = os.path.join(PROJECT_ROOT, "configs", "svm_best.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy {config_path}. Hãy chạy tune_svm.py trước.")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    print("=" * 60)
    print("  SVM FINAL TRAINING — Best Config")
    print("=" * 60)
    print(f"  C            : {config['C']}")
    print(f"  ngram_range  : {tuple(config['ngram_range'])}")
    print(f"  max_features : {config['max_features']}")
    print(f"  sublinear_tf : {config['sublinear_tf']}")
    print(f"  Dev best F1  : {config['dev_f1']*100:.2f}%")
    print()

    # 2. Load data
    data_dir = find_data_dir()
    print(f"Load dataset from: {data_dir}")
    splits = load_dataset(data_dir)

    X_train, y_train = splits["train"]
    X_dev, y_dev = splits["dev"]
    X_test, y_test = splits["test"]

    # 3. Merge train + dev
    X_train_dev = X_train + X_dev
    y_train_dev = list(y_train) + list(y_dev)
    print(f"\nTrain+Dev size: {len(X_train_dev)} mẫu")
    print(f"Test size     : {len(X_test)} mẫu")

    # 4. Preprocess
    print("\nPreprocess text...")
    preprocessor = Preprocessor()
    X_train_dev_clean = preprocessor.clean_batch(X_train_dev)
    X_test_clean = preprocessor.clean_batch(X_test)

    # 5. Train final model
    print("\nTrain SVM with best params...")
    model = SVMClassifier(
        C=config["C"],
        ngram_range=tuple(config["ngram_range"]),
        max_features=config["max_features"],
        sublinear_tf=config["sublinear_tf"],
    )
    model.fit(X_train_dev_clean, y_train_dev)

    # 6. Evaluate on test
    print("\nEvaluate on test set...")
    result = evaluate(model, X_test_clean, y_test)
    print_result_full(result)

    # 7. Save model
    model_path = os.path.join(PROJECT_ROOT, "models", "svm_best.joblib")
    save_model(model, model_path)
    print(f"\nSaved final model to: {model_path}")


if __name__ == "__main__":
    main()
