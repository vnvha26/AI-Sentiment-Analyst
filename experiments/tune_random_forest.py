import io
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from algorithms.random_forest import RandomForestSentiment
from data.loader import load_dataset
from evaluation.evaluator import evaluate, print_result
from preprocessing.preprocessor import Preprocessor


DATA_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "uit-vsfc-sentiment"),
    os.path.join(PROJECT_ROOT, "data", "data", "uit-vsfc-sentiment"),
]


PARAM_GRID = [
    {
        "n_estimators": 50,
        "max_depth": None,
        "ngram_range": (1, 1),
        "max_features": 10000,
        "sublinear_tf": True,
    },
    {
        "n_estimators": 100,
        "max_depth": None,
        "ngram_range": (1, 1),
        "max_features": 10000,
        "sublinear_tf": True,
    },
    {
        "n_estimators": 100,
        "max_depth": 50,
        "ngram_range": (1, 1),
        "max_features": 10000,
        "sublinear_tf": True,
    },
    {
        "n_estimators": 100,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
    },
]


def find_data_dir():
    for data_dir in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(data_dir, "train.csv")):
            return data_dir
    return DATA_DIR_CANDIDATES[0]


def print_row(index, params, result):
    print(
        f"{index:<3} "
        f"{params['n_estimators']:<12} "
        f"{str(params['max_depth']):<10} "
        f"{str(params['ngram_range']):<12} "
        f"{params['max_features']:<12} "
        f"{str(params['sublinear_tf']):<12} "
        f"{result['accuracy'] * 100:>8.2f}% "
        f"{result['f1'] * 100:>8.2f}%"
    )


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

    print("\nTune Random Forest on dev set")
    print(
        f"{'No':<3} {'n_estimators':<12} {'max_depth':<10} "
        f"{'ngram':<12} {'max_features':<12} {'sublinear_tf':<12} "
        f"{'Accuracy':>9} {'F1':>9}"
    )
    print("-" * 89)

    best = None
    for index, params in enumerate(PARAM_GRID, start=1):
        model = RandomForestSentiment(**params)
        model.fit(X_train_clean, y_train)
        result = evaluate(model, X_dev_clean, y_dev)
        print_row(index, params, result)

        if best is None or result["f1"] > best["result"]["f1"]:
            best = {
                "params": params,
                "result": result,
            }

    print("\nBest config by dev F1:")
    print(best["params"])
    print_result(best["result"])
    print("\nSau bước này mới dùng config tốt nhất để train final và đánh giá trên test.")


if __name__ == "__main__":
    main()
