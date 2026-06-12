from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "name": model.name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_time": model.train_time,
    }


def print_result(result):
    print("\n=== Evaluation Result ===")
    print(f"Model    : {result['name']}")
    print(f"Accuracy : {result['accuracy'] * 100:.2f}%")
    print(f"F1-score : {result['f1'] * 100:.2f}%")
    print(f"Train(s) : {result['train_time']:.3f}")
