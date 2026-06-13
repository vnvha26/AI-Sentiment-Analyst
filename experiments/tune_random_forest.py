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

NEUTRAL_LABEL = 1

MODEL_PARAM_KEYS = {
    "n_estimators",
    "max_depth",
    "ngram_range",
    "max_features",
    "sublinear_tf",
    "min_samples_leaf",
    "class_weight",
}


PARAM_GRID = [
    {
        "n_estimators": 100,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
        "min_samples_leaf": 1,
        "class_weight": None,
        "neutral_multiplier": 1,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
        "min_samples_leaf": 1,
        "class_weight": None,
        "neutral_multiplier": 1,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "min_samples_leaf": 1,
        "class_weight": None,
        "neutral_multiplier": 1,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": False,
        "min_samples_leaf": 1,
        "class_weight": None,
        "neutral_multiplier": 1,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "neutral_multiplier": 1,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "neutral_multiplier": 1,
    },
    {
        "n_estimators": 200,
        "max_depth": 100,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "neutral_multiplier": 1,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "neutral_multiplier": 1,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
        "min_samples_leaf": 1,
        "class_weight": None,
        "neutral_multiplier": 3,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
        "min_samples_leaf": 1,
        "class_weight": None,
        "neutral_multiplier": 5,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
        "min_samples_leaf": 1,
        "class_weight": None,
        "neutral_multiplier": 8,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "neutral_multiplier": 3,
    },
]


def find_data_dir():
    for data_dir in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(data_dir, "train.csv")):
            return data_dir
    return DATA_DIR_CANDIDATES[0]


def get_label_f1(result, label_id):
    for item in result.get("label_scores", []):
        if item["label"] == label_id:
            return item.get("f1", 0)
    return 0


def get_model_params(params):
    return {
        key: value
        for key, value in params.items()
        if key in MODEL_PARAM_KEYS
    }


def oversample_label(X, y, label, multiplier):
    if multiplier <= 1:
        return X, y

    X_resampled = list(X)
    y_resampled = list(y)

    for text, target in zip(X, y):
        if target == label:
            for _ in range(multiplier - 1):
                X_resampled.append(text)
                y_resampled.append(target)

    return X_resampled, y_resampled


def print_row(index, params, result):
    neutral_f1 = get_label_f1(result, NEUTRAL_LABEL)
    print(
        f"{index:<3} "
        f"{params['n_estimators']:<6} "
        f"{str(params['max_depth']):<8} "
        f"{str(params['ngram_range']):<8} "
        f"{params['max_features']:<8} "
        f"{str(params['sublinear_tf']):<6} "
        f"{params['min_samples_leaf']:<4} "
        f"{str(params['class_weight']):<10} "
        f"{params['neutral_multiplier']:<5} "
        f"{result['accuracy'] * 100:>8.2f}% "
        f"{result['f1'] * 100:>8.2f}% "
        f"{neutral_f1 * 100:>10.2f}%"
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
        f"{'No':<3} {'trees':<6} {'depth':<8} {'ngram':<8} "
        f"{'max_feat':<8} {'subtf':<6} {'leaf':<4} {'weight':<10} "
        f"{'neu_x':<5} "
        f"{'Accuracy':>9} {'F1':>9} {'Neutral F1':>11}"
    )
    print("-" * 106)

    best = None
    best_neutral = None
    for index, params in enumerate(PARAM_GRID, start=1):
        X_fit, y_fit = oversample_label(
            X_train_clean,
            y_train,
            NEUTRAL_LABEL,
            params["neutral_multiplier"],
        )
        model = RandomForestSentiment(**get_model_params(params))
        model.fit(X_fit, y_fit)
        result = evaluate(model, X_dev_clean, y_dev)
        print_row(index, params, result)

        if best is None or result["f1"] > best["result"]["f1"]:
            best = {
                "params": params,
                "result": result,
            }

        if (
            best_neutral is None
            or get_label_f1(result, NEUTRAL_LABEL)
            > get_label_f1(best_neutral["result"], NEUTRAL_LABEL)
        ):
            best_neutral = {
                "params": params,
                "result": result,
            }

    print("\nBest config by dev F1:")
    print(best["params"])
    print_result(best["result"])

    if best_neutral["params"] != best["params"]:
        print("\nBest config by Neutral F1:")
        print(best_neutral["params"])
        print(
            "Neutral F1: "
            f"{get_label_f1(best_neutral['result'], NEUTRAL_LABEL) * 100:.2f}%"
        )

    print("\nNext step: train final with the selected config, then evaluate once on test.")


if __name__ == "__main__":
    main()
