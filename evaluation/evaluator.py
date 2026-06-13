from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    labels = list(LABEL_NAMES)

    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    weighted_precision, weighted_recall, _, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "name": model.name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision": weighted_precision,
        "recall": weighted_recall,
        "train_time": model.train_time,
        "label_scores": _label_scores(
            y_test,
            y_pred,
            precision,
            recall,
            f1_per_class,
            support,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "y_true": y_test,
        "y_pred": y_pred,
    }


def _label_scores(y_true, y_pred, precision, recall, f1_per_class, support):
    scores = []

    for index, (label, label_name) in enumerate(LABEL_NAMES.items()):
        total = int(support[index])
        correct = sum(
            1
            for true_label, pred_label in zip(y_true, y_pred)
            if true_label == label and pred_label == label
        )

        scores.append({
            "label": label,
            "name": label_name,
            "correct": correct,
            "total": total,
            "percent": correct / total if total else 0,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1_per_class[index]),
        })

    return scores


def print_result(result):
    _print_main_metrics(result)
    _print_label_scores(result)
    _print_classification_report(result)


def print_result_full(result):
    _print_main_metrics(result)
    _print_label_scores(result)
    print_confusion_matrix(result)
    _print_classification_report(result)


def print_comparison_table(results):
    if not results:
        print("\nNo evaluation results.")
        return

    print("\n" + "=" * 78)
    print("  Model Comparison")
    print("=" * 78)
    print(
        f"  {'Model':<24} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>8} {'F1':>8} {'Train(s)':>10}"
    )
    print("-" * 78)

    for result in results:
        print(
            f"  {result['name']:<24} "
            f"{result['accuracy'] * 100:>8.2f}% "
            f"{result.get('precision', 0) * 100:>9.2f}% "
            f"{result.get('recall', 0) * 100:>7.2f}% "
            f"{result['f1'] * 100:>7.2f}% "
            f"{result['train_time']:>9.3f}s"
        )

    print("=" * 78)

    best = max(results, key=lambda item: item["f1"])
    fastest = min(results, key=lambda item: item["train_time"])
    print(f"\n  Best F1       : {best['name']} ({best['f1'] * 100:.2f}%)")
    print(f"  Fastest train : {fastest['name']} ({fastest['train_time']:.3f}s)")


def print_confusion_matrix(result):
    cm = result.get("confusion_matrix")
    if cm is None:
        print("\nNo confusion matrix.")
        return

    labels = [LABEL_NAMES[label] for label in LABEL_NAMES]
    print(f"\nConfusion Matrix - {result['name']}")
    print(f"{'':>12}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print()

    for index, row in enumerate(cm):
        print(f"{labels[index]:>12}", end="")
        for value in row:
            print(f"{value:>10}", end="")
        print()


def _print_main_metrics(result):
    print("\n=== Evaluation Result ===")
    print(f"Model    : {result['name']}")
    print(f"Accuracy : {result['accuracy'] * 100:.2f}%")
    print(f"Precision: {result.get('precision', 0) * 100:.2f}%")
    print(f"Recall   : {result.get('recall', 0) * 100:.2f}%")
    print(f"F1-score : {result['f1'] * 100:.2f}%")
    print(f"Train(s) : {result['train_time']:.3f}")


def _print_label_scores(result):
    print("\nPer-label metrics:")
    print(
        f"{'Label':<10} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>8} {'F1':>8} {'Correct/Total':>15}"
    )
    print("-" * 66)

    for item in result["label_scores"]:
        print(
            f"{item['name']:<10} "
            f"{item['percent'] * 100:>8.2f}% "
            f"{item.get('precision', 0) * 100:>9.2f}% "
            f"{item.get('recall', 0) * 100:>7.2f}% "
            f"{item.get('f1', 0) * 100:>7.2f}% "
            f"{item['correct']:>7}/{item['total']:<7}"
        )


def _print_classification_report(result):
    y_true = result.get("y_true")
    y_pred = result.get("y_pred")
    if y_true is None or y_pred is None or len(y_true) == 0:
        return

    print()
    print(classification_report(
        y_true,
        y_pred,
        labels=list(LABEL_NAMES),
        target_names=[LABEL_NAMES[label] for label in LABEL_NAMES],
        zero_division=0,
    ))
