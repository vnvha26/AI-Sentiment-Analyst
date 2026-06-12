from sklearn.metrics import accuracy_score, f1_score


LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "name": model.name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_time": model.train_time,
    }


        "label_scores": _label_scores(y_test, y_pred),
    }sS


def _label_scores(y_true, y_pred):
    scores = []

    for label, label_name in LABEL_NAMES.items():
        total = 0
        correct = 0

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label == label:
                total += 1
                if pred_label == label:
                    correct += 1

        percent = correct / total if total > 0 else 0
        scores.append({
            "label": label,
            "name": label_name,
            "correct": correct,
            "total": total,
            "percent": percent,
        })

    return scores


def print_result(result):
    print("\n=== Evaluation Result ===")
    print(f"Model    : {result['name']}")
    print(f"Accuracy : {result['accuracy'] * 100:.2f}%")
    print(f"F1-score : {result['f1'] * 100:.2f}%")
    print(f"Train(s) : {result['train_time']:.3f}")

    print("\nPer-label accuracy:")
    for item in result["label_scores"]:
        print(
            f"- {item['name']:<8}: "
            f"{item['percent'] * 100:>6.2f}% "
            f"({item['correct']}/{item['total']})"
        )
