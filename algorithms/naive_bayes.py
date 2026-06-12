import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.name = "Naive Bayes"
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
        )

        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.train_time = 0

    def fit(self, X_train, y_train):
        start = time.time()

        X_vec = self.vectorizer.fit_transform(X_train)
        y = np.asarray(y_train)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X_vec.shape[1]

        class_count = np.zeros(n_classes, dtype=float)
        feature_count = np.zeros((n_classes, n_features), dtype=float)

        for class_index, class_label in enumerate(self.classes_):
            rows = y == class_label
            class_count[class_index] = rows.sum()
            feature_count[class_index] = np.asarray(X_vec[rows].sum(axis=0)).ravel()

        self.class_log_prior_ = np.log(class_count / class_count.sum())

        smoothed_feature_count = feature_count + self.alpha
        smoothed_class_total = smoothed_feature_count.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(
            smoothed_feature_count / smoothed_class_total
        )

        self.train_time = time.time() - start
        print(f"[{self.name}] Train done in {self.train_time:.3f}s")

    def predict(self, X):
        scores = self._joint_log_likelihood(X)
        class_indices = np.argmax(scores, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        scores = self._joint_log_likelihood(X)
        return self._softmax(scores)

    def predict_one(self, text):
        proba = self.predict_proba([text])[0]
        class_index = int(proba.argmax())
        label = int(self.classes_[class_index])
        confidence = float(proba[class_index])
        return label, confidence

    def _joint_log_likelihood(self, X):
        self._check_is_fitted()
        X_vec = self.vectorizer.transform(X)
        return X_vec @ self.feature_log_prob_.T + self.class_log_prior_

    def _check_is_fitted(self):
        if self.classes_ is None or self.feature_log_prob_ is None:
            raise RuntimeError("Model chưa được train. Hãy gọi fit() trước.")

    @staticmethod
    def _softmax(scores):
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
