import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class LogisticRegressionSentiment:
    def __init__(
        self,
        C=1.0,
        max_iter=1000,
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        class_weight=None,
        random_state=42,
    ):
        self.name = "Logistic Regression"
        self.vectorizer_params = {
            "ngram_range": ngram_range,
            "max_features": max_features,
            "sublinear_tf": sublinear_tf,
        }
        self.model_params = {
            "C": C,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "solver": "lbfgs",
            "random_state": random_state,
        }
        self.params = {
            **self.vectorizer_params,
            **self.model_params,
        }
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        self.model = LogisticRegression(**self.model_params)
        self.train_time = 0

    def fit(self, X_train, y_train):
        start = time.time()
        X_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_vec, y_train)
        self.train_time = time.time() - start
        print(f"[{self.name}] Train done in {self.train_time:.3f}s")

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)

    def predict_one(self, text):
        proba = self.predict_proba([text])[0]
        label = int(proba.argmax())
        confidence = float(proba[label])
        return label, confidence

    def top_features(self, n=15):
        if not hasattr(self.model, "coef_"):
            raise RuntimeError(
                "Model is not trained. Call fit() before top_features()."
            )

        vocab = self.vectorizer.get_feature_names_out()
        class_features = {}

        print(f"\nTop {n} positive features per class ({self.name}):")
        for class_index, class_label in enumerate(self.model.classes_):
            weights = self.model.coef_[class_index]
            top_indices = weights.argsort()[-n:][::-1]

            features = [
                {
                    "feature": vocab[index],
                    "weight": float(weights[index]),
                }
                for index in top_indices
            ]
            class_features[int(class_label)] = features

            print(f"\nClass {int(class_label)}:")
            for rank, item in enumerate(features, start=1):
                print(
                    f"  {rank:>2}. "
                    f"{item['feature']:<20} "
                    f"{item['weight']:.5f}"
                )

        return class_features
