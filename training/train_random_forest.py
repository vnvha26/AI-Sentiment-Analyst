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
    data_dir = find_data_dir()
    print(f"Load dataset from: {data_dir}")
    splits = load_dataset(data_dir)

    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]

    print("\nPreprocess text...")
    preprocessor = Preprocessor()
    X_train_clean = preprocessor.clean_batch(X_train)
    X_test_clean = preprocessor.clean_batch(X_test)

    print("\nTrain Random Forest...")
    model = RandomForestSentiment(n_estimators=50)
    model.fit(X_train_clean, y_train)

    print("\nEvaluate on test set...")
    result = evaluate(model, X_test_clean, y_test)
    print_result(result)

    model_path = os.path.join(PROJECT_ROOT, "models", "random_forest.joblib")
    save_model(model, model_path)


if __name__ == "__main__":
    main()
