import io
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from algorithms.logistic_regression import LogisticRegressionSentiment
from data.loader import load_dataset
from evaluation.evaluator import evaluate, print_result
from preprocessing.preprocessor import Preprocessor
from utils.model_manager import save_model


DATA_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "uit-vsfc-sentiment"),
    os.path.join(PROJECT_ROOT, "data", "data", "uit-vsfc-sentiment"),
]

BEST_PARAMS = {
    "C": 2.0,
    "max_iter": 1000,
    "ngram_range": (1, 2),
    "max_features": 30000,
    "sublinear_tf": True,
    "class_weight": "balanced",
}


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
    X_test, y_test = splits["test"]

    print("\nPreprocess text...")
    preprocessor = Preprocessor()
    X_train_clean = preprocessor.clean_batch(X_train)
    X_test_clean = preprocessor.clean_batch(X_test)

    print("\nTrain final Logistic Regression...")
    print(f"Best params from dev tuning: {BEST_PARAMS}")
    model = LogisticRegressionSentiment(**BEST_PARAMS)
    model.fit(X_train_clean, y_train)

    print("\nEvaluate on test set...")
    result = evaluate(model, X_test_clean, y_test)
    print_result(result)

    model.top_features(15)

    model_path = os.path.join(PROJECT_ROOT, "models", "logistic_regression.joblib")
    save_model(model, model_path)


if __name__ == "__main__":
    main()
