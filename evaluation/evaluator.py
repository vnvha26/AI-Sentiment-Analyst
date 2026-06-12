from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

LABELS_VI = ["Tiêu cực", "Trung lập", "Tích cực"]


def evaluate(model, X_test, y_test):
    """
    Đánh giá 1 model trên tập test.
    Trả về dict chứa các metrics.
    Giữ nguyên tất cả key cũ (accuracy, f1, train_time, label_scores)
    và thêm precision, recall, confusion_matrix.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    return {
        "name": model.name,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "train_time": model.train_time,
        "label_scores": _label_scores(y_test, y_pred),
        "confusion_matrix": cm,
        "y_true": y_test,
        "y_pred": y_pred,
    }


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
    """
    In kết quả evaluate cơ bản (backward-compatible với code cũ).
    """
    print("\n=== Evaluation Result ===")
    print(f"Model    : {result['name']}")
    print(f"Accuracy : {result['accuracy'] * 100:.2f}%")
    print(f"F1-score : {result['f1'] * 100:.2f}%")
    print(f"Train(s) : {result['train_time']:.3f}")

    # In precision/recall nếu có (không bắt buộc để tránh lỗi với dict cũ)
    if "precision" in result:
        print(f"Precision: {result['precision'] * 100:.2f}%")
    if "recall" in result:
        print(f"Recall   : {result['recall'] * 100:.2f}%")

    print("\nPer-label accuracy:")
    for item in result["label_scores"]:
        print(
            f"- {item['name']:<8}: "
            f"{item['percent'] * 100:>6.2f}% "
            f"({item['correct']}/{item['total']})"
        )

    # In classification report nếu y_true và y_pred có sẵn
    y_true = result.get("y_true")
    y_pred = result.get("y_pred")
    if y_true is not None and y_pred is not None and len(y_true) > 0:
        print()
        print(classification_report(
            y_true,
            y_pred,
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
            zero_division=0,
        ))


def print_result_full(result):
    """
    In kết quả đầy đủ với precision, recall, confusion_matrix.
    """
    print("\n=== Evaluation Result ===")
    print(f"Model    : {result['name']}")
    print(f"Accuracy : {result['accuracy'] * 100:.2f}%")
    print(f"Precision: {result.get('precision', 0) * 100:.2f}%")
    print(f"Recall   : {result.get('recall', 0) * 100:.2f}%")
    print(f"F1-score : {result['f1'] * 100:.2f}%")
    print(f"Train(s) : {result['train_time']:.3f}")

    print("\nPer-label accuracy:")
    for item in result["label_scores"]:
        print(
            f"- {item['name']:<8}: "
            f"{item['percent'] * 100:>6.2f}% "
            f"({item['correct']}/{item['total']})"
        )

    print_confusion_matrix(result)

    if "y_pred" in result and "y_true" in result:
        print()
        print(classification_report(
            result["y_true"],
            result["y_pred"],
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
            zero_division=0,
        ))


def print_comparison_table(results):
    """In bảng so sánh tất cả thuật toán"""
    print("\n" + "=" * 72)
    print("  BẢNG SO SÁNH CÁC THUẬT TOÁN")
    print("=" * 72)
    print(
        f"  {'Thuật toán':<22} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>8} {'F1':>8} {'Train(s)':>10}"
    )
    print("-" * 72)
    for r in results:
        print(
            f"  {r['name']:<22} {r['accuracy'] * 100:>8.2f}% "
            f"{r.get('precision', 0) * 100:>9.2f}% "
            f"{r.get('recall', 0) * 100:>7.2f}% "
            f"{r['f1'] * 100:>7.2f}% "
            f"{r['train_time']:>9.3f}s"
        )
    print("=" * 72)

    best = max(results, key=lambda r: r["f1"])
    fastest = min(results, key=lambda r: r["train_time"])
    print(f"\n  🏆 F1 cao nhất  : {best['name']} ({best['f1']*100:.2f}%)")
    print(f"  ⚡ Train nhanh nhất: {fastest['name']} ({fastest['train_time']:.3f}s)")


def print_confusion_matrix(result):
    """In confusion matrix dạng text"""
    cm = result.get("confusion_matrix")
    if cm is None:
        print("\n  [Không có confusion_matrix]")
        return

    name = result["name"]
    labels = [LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]]
    print(f"\n  Confusion Matrix — {name}")
    print(f"  {'':15}", end="")
    for label in labels:
        print(f"  {label:>10}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {labels[i]:>15}", end="")
        for val in row:
            print(f"  {val:>10}", end="")
        print()
