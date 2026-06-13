import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


class RandomForestSentiment:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        ngram_range=(1, 1),
        max_features=10000,
        sublinear_tf=True,
        random_state=42,
    ):
        self.name = "Random Forest"
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
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
