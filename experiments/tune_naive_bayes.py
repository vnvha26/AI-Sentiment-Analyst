import io
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from algorithms.naive_bayes import NaiveBayesClassifier
from data.loader import load_dataset
from evaluation.evaluator import evaluate, print_result
from preprocessing.preprocessor import Preprocessor
from utils.imbalance import NEUTRAL_LABEL, oversample_label, sample_weight_for_label


DATA_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "uit-vsfc-sentiment"),
    os.path.join(PROJECT_ROOT, "data", "data", "uit-vsfc-sentiment"),
]
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "naive_bayes_best.json")

MODEL_PARAM_KEYS = {
    "alpha",
    "ngram_range",
    "max_features",
    "sublinear_tf",
    "fit_prior",
    "class_prior",
}

PARAM_GRID = [
    {
        "alpha": 0.1,
        "ngram_range": (1, 1),
        "max_features": 15000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 1,
    },
    {
        "alpha": 0.5,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 1,
    },
    {
        "alpha": 1.0,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 1,
    },
    {
        "alpha": 2.0,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 1,
    },
    {
        "alpha": 1.0,
        "ngram_range": (1, 2),
        "max_features": 50000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 1,
    },
    {
        "alpha": 1.0,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": False,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 1,
    },
    {
        "alpha": 1.0,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": False,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 1,
    },
    {
        "alpha": 0.5,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 3,
        "neutral_weight": 1,
    },
    {
        "alpha": 1.0,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 3,
        "neutral_weight": 1,
    },
    {
        "alpha": 0.5,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 2,
    },
    {
        "alpha": 0.5,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 3,
    },
    {
        "alpha": 0.5,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 5,
    },
    {
        "alpha": 0.3,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 4,
    },
    {
        "alpha": 0.3,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 5,
    },
    {
        "alpha": 0.3,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 6,
    },
    {
        "alpha": 0.5,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 4,
    },
    {
        "alpha": 0.5,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 6,
    },
    {
        "alpha": 0.7,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 4,
    },
    {
        "alpha": 0.7,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 5,
    },
    {
        "alpha": 0.7,
        "ngram_range": (1, 2),
        "max_features": 30000,
        "sublinear_tf": True,
        "fit_prior": True,
        "class_prior": None,
        "neutral_multiplier": 1,
        "neutral_weight": 6,
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


def get_macro_f1(result):
    label_scores = result.get("label_scores", [])
    if not label_scores:
        return 0
    return sum(item.get("f1", 0) for item in label_scores) / len(label_scores)


def get_model_params(params):
    return {
        key: value
        for key, value in params.items()
        if key in MODEL_PARAM_KEYS
    }


def save_best_config(best):
    config = dict(best["params"])
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    return CONFIG_PATH


def print_row(index, params, result):
    neutral_f1 = get_label_f1(result, NEUTRAL_LABEL)
    macro_f1 = get_macro_f1(result)
    print(
        f"{index:<3} "
        f"{params['alpha']:<5} "
        f"{str(params['ngram_range']):<8} "
        f"{params['max_features']:<8} "
        f"{str(params['sublinear_tf']):<6} "
        f"{str(params['fit_prior']):<7} "
        f"{params['neutral_multiplier']:<5} "
        f"{params['neutral_weight']:<5} "
        f"{result['accuracy'] * 100:>8.2f}% "
        f"{result['f1'] * 100:>8.2f}% "
        f"{macro_f1 * 100:>9.2f}% "
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

    print("\nTune Naive Bayes on dev set")
    print(
        f"{'No':<3} {'alpha':<5} {'ngram':<8} {'max_feat':<8} "
        f"{'subtf':<6} {'prior':<7} {'neu_x':<5} {'neu_w':<5} "
        f"{'Accuracy':>9} {'F1':>9} {'Macro F1':>10} {'Neutral F1':>11}"
    )
    print("-" * 103)
    print("Selection metric: weighted dev F1")
    print("Macro F1 and Neutral F1 are reported separately to analyze class imbalance.\n")

    best = None
    best_macro = None
    best_neutral = None
    for index, params in enumerate(PARAM_GRID, start=1):
        X_fit, y_fit = oversample_label(
            X_train_clean,
            y_train,
            NEUTRAL_LABEL,
            params["neutral_multiplier"],
        )
        sample_weight = sample_weight_for_label(
            y_fit,
            NEUTRAL_LABEL,
            params["neutral_weight"],
        )
        model = NaiveBayesClassifier(**get_model_params(params))
        model.fit(X_fit, y_fit, sample_weight=sample_weight)
        result = evaluate(model, X_dev_clean, y_dev)
        print_row(index, params, result)

        if best is None or result["f1"] > best["result"]["f1"]:
            best = {
                "params": params,
                "result": result,
            }

        if best_macro is None or get_macro_f1(result) > get_macro_f1(best_macro["result"]):
            best_macro = {
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

    if best_macro["params"] != best["params"]:
        print("\nBest config by Macro F1:")
        print(best_macro["params"])
        print(f"Macro F1: {get_macro_f1(best_macro['result']) * 100:.2f}%")

    if best_neutral["params"] != best["params"]:
        print("\nBest config by Neutral F1:")
        print(best_neutral["params"])
        print(
            "Neutral F1: "
            f"{get_label_f1(best_neutral['result'], NEUTRAL_LABEL) * 100:.2f}%"
        )

    config_path = save_best_config(best)
    print(f"\nSaved best config to: {config_path}")
    print("\nNext step: train final with the selected config, then evaluate once on test.")


if __name__ == "__main__":
    main()
