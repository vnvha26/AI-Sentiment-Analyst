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
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )
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
