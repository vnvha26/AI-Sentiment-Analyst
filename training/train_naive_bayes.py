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
from utils.imbalance import NEUTRAL_LABEL, oversample_label
from utils.model_manager import save_model


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

# Selected from experiments/tune_naive_bayes.py by weighted dev F1.
# Neutral F1 is tracked separately to analyze class imbalance.
DEFAULT_CONFIG = {
    "alpha": 0.5,
    "ngram_range": (1, 2),
    "max_features": 30000,
    "sublinear_tf": True,
    "fit_prior": True,
    "class_prior": None,
    "neutral_multiplier": 3,
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

    print("\nTrain final Naive Bayes...")
    print(f"Best params from dev tuning: {model_params}")
    print(f"Neutral oversampling multiplier: {neutral_multiplier}")
    if "dev_f1" in config:
        print(f"Dev F1 from tuning: {config['dev_f1'] * 100:.2f}%")
    X_fit, y_fit = oversample_label(
        X_train_clean,
        y_train,
        NEUTRAL_LABEL,
        neutral_multiplier,
    )
    model = NaiveBayesClassifier(**model_params)
    model.fit(X_fit, y_fit)

    print("\nEvaluate on test set...")
    result = evaluate(model, X_test_clean, y_test)
    print_result(result)

    model.top_features(15)

    model_path = os.path.join(PROJECT_ROOT, "models", "naive_bayes.joblib")
    save_model(model, model_path)


if __name__ == "__main__":
    main()
