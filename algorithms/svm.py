"""
svm.py — Support Vector Machine Classifier

Ý tưởng:
  Tìm siêu phẳng (hyperplane) tối đa hóa margin giữa các class.
  Dùng kernel trick để xử lý không gian phi tuyến.
  Với text thường dùng kernel tuyến tính vì features đã rất nhiều chiều.

Độ phức tạp:
  - Training  : O(N^2) đến O(N^3) với kernel — dùng LinearSVC: O(N * V)
  - Prediction: O(V * C)

Khi nào dùng:
  - Dataset vừa đến lớn
  - Số chiều features cao (text rất phù hợp)
  - Cần accuracy cao, chấp nhận train chậm hơn

Ưu điểm: thường cho accuracy cao nhất trong các model tuyến tính với text
Nhược điểm: không cho xác suất trực tiếp, chậm với dataset rất lớn
"""

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
import time


class SVMClassifier:
    def __init__(
        self,
        C=1.0,
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        class_weight=None,
        max_iter=2000,
    ):
        """
        Args:
            C: regularization parameter
            ngram_range: tuple (min_n, max_n) for TF-IDF n-grams
            max_features: max number of TF-IDF features
            sublinear_tf: apply sublinear tf scaling (1 + log(tf))
            class_weight: 'balanced' or None
            max_iter: max iterations for LinearSVC
        """
        self.name = "SVM (LinearSVC)"
        self.C = C
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.class_weight = class_weight
        self.max_iter = max_iter

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        # CalibratedClassifierCV để có predict_proba
        base_svm = LinearSVC(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
        )
        self.model = CalibratedClassifierCV(base_svm, cv=3)
        self.train_time = 0

    def fit(self, X_train, y_train):
        t0 = time.time()
        X_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_vec, y_train)
        self.train_time = time.time() - t0
        print(f"  [{self.name}] Train xong trong {self.train_time:.3f}s")

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
