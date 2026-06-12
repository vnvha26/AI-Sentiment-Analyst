import os

import joblib


def save_model(model, path="models/naive_bayes.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to: {path}")


def load_model(path="models/naive_bayes.joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    return joblib.load(path)


def model_exists(path="models/naive_bayes.joblib"):
    return os.path.exists(path)
