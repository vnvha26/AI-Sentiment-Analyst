import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class NaiveBayesClassifier:
    def __init__(
        self,
        alpha=1.0,
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        fit_prior=True,
        class_prior=None,
    ):
        self.name = "Naive Bayes"
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.vectorizer_params = {
            "ngram_range": ngram_range,
            "max_features": max_features,
            "sublinear_tf": sublinear_tf,
        }
        self.params = {
            "alpha": alpha,
            "fit_prior": fit_prior,
            "class_prior": class_prior,
            **self.vectorizer_params,
        }
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)

        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.train_time = 0

    def fit(self, X_train, y_train, sample_weight=None):
        start = time.time()

        X_vec = self.vectorizer.fit_transform(X_train)
        y = np.asarray(y_train)
        weights = self._validate_sample_weight(sample_weight, y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X_vec.shape[1]

        class_count = np.zeros(n_classes, dtype=float)
        feature_count = np.zeros((n_classes, n_features), dtype=float)

        for class_index, class_label in enumerate(self.classes_):
            rows = y == class_label
            class_weights = weights[rows]
            class_count[class_index] = class_weights.sum()
            weighted_x = X_vec[rows].multiply(class_weights[:, np.newaxis])
            feature_count[class_index] = np.asarray(weighted_x.sum(axis=0)).ravel()

        self.class_log_prior_ = self._estimate_class_log_prior(class_count)

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

    def top_features(self, n=15):
        self._check_is_fitted()
        vocab = self.vectorizer.get_feature_names_out()
        class_features = {}

        print(f"\nTop {n} probable features per class ({self.name}):")
        for class_index, class_label in enumerate(self.classes_):
            scores = self.feature_log_prob_[class_index]
            top_indices = scores.argsort()[-n:][::-1]

            features = [
                {
                    "feature": vocab[index],
                    "log_prob": float(scores[index]),
                }
                for index in top_indices
            ]
            class_features[int(class_label)] = features

            print(f"\nClass {int(class_label)}:")
            for rank, item in enumerate(features, start=1):
                print(
                    f"  {rank:>2}. "
                    f"{item['feature']:<20} "
                    f"{item['log_prob']:.5f}"
                )

        return class_features

    def _joint_log_likelihood(self, X):
        self._check_is_fitted()
        X_vec = self.vectorizer.transform(X)
        return X_vec @ self.feature_log_prob_.T + self.class_log_prior_

    def _check_is_fitted(self):
        if self.classes_ is None or self.feature_log_prob_ is None:
            raise RuntimeError("Model is not trained. Call fit() before prediction.")

    def _validate_sample_weight(self, sample_weight, y):
        if sample_weight is None:
            return np.ones(len(y), dtype=float)

        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape[0] != len(y):
            raise ValueError("sample_weight must have the same length as y_train.")
        return weights

    def _estimate_class_log_prior(self, class_count):
        if self.class_prior is not None:
            priors = np.asarray(self.class_prior, dtype=float)
            if priors.shape[0] != len(self.classes_):
                raise ValueError("class_prior must match the number of classes.")
            priors = priors / priors.sum()
            return np.log(priors)

        if not self.fit_prior:
            return np.full(len(self.classes_), -np.log(len(self.classes_)))

        return np.log(class_count / class_count.sum())

    @staticmethod
    def _softmax(scores):
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
