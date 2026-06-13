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
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=None,
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
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
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

    def top_features(self, n=15):
        vocab = self.vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_
        top_indices = importances.argsort()[-n:][::-1]

        features = [
            {
                "feature": vocab[index],
                "importance": float(importances[index]),
            }
            for index in top_indices
        ]

        print(f"\nTop {n} important features ({self.name}):")
        for rank, item in enumerate(features, start=1):
            print(
                f"  {rank:>2}. "
                f"{item['feature']:<20} "
                f"{item['importance']:.5f}"
            )

        return features
