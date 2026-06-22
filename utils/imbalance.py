from collections import Counter


NEUTRAL_LABEL = 1


def label_distribution(y):
    return dict(Counter(y))


def print_label_distribution(y, title="Label distribution"):
    counts = label_distribution(y)
    total = sum(counts.values())

    print(f"\n{title}")
    for label in sorted(counts):
        percent = counts[label] / total * 100 if total else 0
        print(f"  Label {label}: {counts[label]} ({percent:.2f}%)")


def oversample_label(X, y, label=NEUTRAL_LABEL, multiplier=1):
    if multiplier <= 1:
        return list(X), list(y)

    X_resampled = list(X)
    y_resampled = list(y)

    for text, target in zip(X, y):
        if target == label:
            for _ in range(multiplier - 1):
                X_resampled.append(text)
                y_resampled.append(target)

    return X_resampled, y_resampled


def oversample_neutral(X, y, multiplier=1):
    return oversample_label(
        X,
        y,
        label=NEUTRAL_LABEL,
        multiplier=multiplier,
    )


def sample_weight_for_label(y, label=NEUTRAL_LABEL, weight=1.0):
    return [
        float(weight) if target == label else 1.0
        for target in y
    ]


def oversample_to_max_class(X, y):
    counts = Counter(y)
    if not counts:
        return list(X), list(y)

    max_count = max(counts.values())
    X_resampled = list(X)
    y_resampled = list(y)

    for label, count in counts.items():
        if count >= max_count:
            continue

        needed = max_count - count
        samples = [text for text, target in zip(X, y) if target == label]

        for index in range(needed):
            X_resampled.append(samples[index % len(samples)])
            y_resampled.append(label)

    return X_resampled, y_resampled
