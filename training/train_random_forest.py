import io
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from algorithms.random_forest import RandomForestSentiment
from data.loader import load_dataset
from evaluation.evaluator import evaluate, print_result
from preprocessing.preprocessor import Preprocessor
from utils.imbalance import NEUTRAL_LABEL, oversample_label, sample_weight_for_label
from utils.model_manager import save_model


DATA_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "uit-vsfc-sentiment"),
    os.path.join(PROJECT_ROOT, "data", "data", "uit-vsfc-sentiment"),
]
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "random_forest_best.json")
MODEL_PARAM_KEYS = {
    "n_estimators",
    "max_depth",
    "ngram_range",
    "max_features",
    "sublinear_tf",
    "min_samples_leaf",
    "class_weight",
}

# Selected from experiments/tune_random_forest.py by weighted dev F1.
# Neutral F1 is tracked separately to analyze class imbalance.
DEFAULT_CONFIG = {
    "n_estimators": 200,
    "max_depth": None,
    "ngram_range": (1, 2),
    "max_features": 15000,
    "sublinear_tf": False,
    "min_samples_leaf": 1,
    "class_weight": None,
    "neutral_multiplier": 1,
    "neutral_weight": 1,
}


def find_data_dir():
    for data_dir in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(data_dir, "train.csv")):
            return data_dir
    return DATA_DIR_CANDIDATES[0]


def load_best_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Config not found, use default config: {CONFIG_PATH}")
        return dict(DEFAULT_CONFIG)

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = json.load(file)

    if "ngram_range" in config:
        config["ngram_range"] = tuple(config["ngram_range"])

    return config


def get_model_params(config):
    return {
        key: value
        for key, value in config.items()
        if key in MODEL_PARAM_KEYS
    }


def main():
    data_dir = find_data_dir()
    print(f"Load dataset from: {data_dir}")
    splits = load_dataset(data_dir)

    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]

    print("\nPreprocess text...")
    preprocessor = Preprocessor()
    X_train_clean = preprocessor.clean_batch(X_train)
    X_test_clean = preprocessor.clean_batch(X_test)

    config = load_best_config()
    model_params = get_model_params(config)
    neutral_multiplier = config.get("neutral_multiplier", 1)
    neutral_weight = config.get("neutral_weight", 1)

    print("\nTrain final Random Forest...")
    print(f"Best params from dev tuning: {model_params}")
    print(f"Neutral oversampling multiplier: {neutral_multiplier}")
    print(f"Neutral sample weight: {neutral_weight}")
    X_fit, y_fit = oversample_label(
        X_train_clean,
        y_train,
        NEUTRAL_LABEL,
        neutral_multiplier,
    )
    sample_weight = sample_weight_for_label(
        y_fit,
        NEUTRAL_LABEL,
        neutral_weight,
    )
    model = RandomForestSentiment(**model_params)
    model.fit(X_fit, y_fit, sample_weight=sample_weight)

    print("\nEvaluate on test set...")
    result = evaluate(model, X_test_clean, y_test)
    print_result(result)

    model.top_features(15)

    model_path = os.path.join(PROJECT_ROOT, "models", "random_forest.joblib")
    save_model(model, model_path)


if __name__ == "__main__":
    main()
