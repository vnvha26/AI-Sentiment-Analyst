import argparse
import io
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from data.loader import LABEL_MAP
from preprocessing.preprocessor import Preprocessor
from utils.model_manager import load_model


MODEL_PATHS = {
    "naive_bayes": os.path.join(PROJECT_ROOT, "models", "naive_bayes.joblib"),
    "random_forest": os.path.join(PROJECT_ROOT, "models", "random_forest.joblib"),
    "logistic_regression": os.path.join(
        PROJECT_ROOT,
        "models",
        "logistic_regression.joblib",
    ),
    "svm": os.path.join(PROJECT_ROOT, "models", "svm.joblib"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["all", *sorted(MODEL_PATHS)],
        default="all",
        help="Model name to load from models/. Default: all available models.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Predict one sentence directly. If omitted, interactive mode is used.",
    )
    return parser.parse_args()


def load_available_models(model_name="all"):
    selected = (
        MODEL_PATHS
        if model_name == "all"
        else {model_name: MODEL_PATHS[model_name]}
    )
    models = {}

    for name, path in selected.items():
        if not os.path.exists(path):
            continue

        models[name] = load_model(path)

    if not models:
        raise FileNotFoundError(
            "No model files found. Train and save at least one model first."
        )

    return models


def predict_text(model, preprocessor, text):
    cleaned = preprocessor.clean(text)

    if hasattr(model, "predict_one"):
        label, confidence = model.predict_one(cleaned)
    else:
        label = int(model.predict([cleaned])[0])
        confidence = None

    return {
        "input": text,
        "cleaned": cleaned,
        "label": int(label),
        "label_name": LABEL_MAP.get(int(label), str(label)),
        "confidence": confidence,
    }


def predict_with_models(models, preprocessor, text):
    return {
        name: predict_text(model, preprocessor, text)
        for name, model in models.items()
    }


def print_prediction(prediction, model_name=None, show_input=True):
    if model_name is not None:
        print(f"\n[{model_name}]")

    if show_input:
        print(f"Input     : {prediction['input']}")
        print(f"Cleaned   : {prediction['cleaned']}")

    print(f"Prediction: {prediction['label_name']} ({prediction['label']})")

    if prediction["confidence"] is not None:
        print(f"Confidence: {prediction['confidence'] * 100:.2f}%")


def print_predictions(predictions):
    first_prediction = next(iter(predictions.values()))
    print(f"Input     : {first_prediction['input']}")
    print(f"Cleaned   : {first_prediction['cleaned']}")

    for model_name, prediction in predictions.items():
        print_prediction(prediction, model_name=model_name, show_input=False)


def interactive_predict(models, preprocessor):
    print("\nType a sentence to predict. Type 'quit' to exit.")

    while True:
        text = input("\nInput sentence: ").strip()
        if text.lower() in {"quit", "exit", "q", ""}:
            break

        predictions = predict_with_models(models, preprocessor, text)
        print_predictions(predictions)


def main():
    args = parse_args()
    models = load_available_models(args.model)
    preprocessor = Preprocessor()

    if args.text:
        predictions = predict_with_models(models, preprocessor, args.text)
        print_predictions(predictions)
    else:
        interactive_predict(models, preprocessor)


if __name__ == "__main__":
    main()
