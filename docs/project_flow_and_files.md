# Flow project và chức năng các file

Tài liệu này mô tả tổng quan project Sentiment Analysis, áp dụng chung cho tất
cả thuật toán trong repo: Naive Bayes, Random Forest, Logistic Regression, SVM
và các thuật toán có thể thêm sau này.

## 1. Mục tiêu project

Project xử lý bài toán phân loại cảm xúc câu tiếng Việt theo 3 nhãn:

```text
0 -> Negative
1 -> Neutral
2 -> Positive
```

Flow tổng quát:

```text
load data
-> preprocess text
-> train model
-> tune params trên dev
-> chọn best config
-> train final
-> evaluate trên test
-> save model
-> predict/demo/UI
```

Quy tắc quan trọng:

```text
train.csv -> dùng để train model
dev.csv   -> dùng để tune/chọn tham số
test.csv  -> dùng để đánh giá cuối cùng, không dùng để tune
```

## 2. Cấu trúc thư mục

```text
AI_Sentiment_Analysis/
|
|-- algorithms/
|-- data/
|-- docs/
|-- evaluation/
|-- experiments/
|-- models/
|-- prediction/
|-- preprocessing/
|-- training/
|-- utils/
|-- main.py
|-- README.md
```

## 3. data/

Thư mục `data/` phụ trách tải, build và load dataset.

### `data/loader.py`

Chức năng:

- định nghĩa label map:

```python
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
```

- tải raw data UIT-VSFC nếu cần
- build CSV từ raw sentences/sentiments
- load 3 split:

```text
train
dev
test
```

Hàm chính:

```python
load_dataset(data_dir)
```

Kết quả trả về:

```python
{
    "train": (X_train, y_train),
    "dev": (X_dev, y_dev),
    "test": (X_test, y_test),
}
```

Tất cả train/tune scripts nên dùng file này để load data.

## 4. preprocessing/

Thư mục `preprocessing/` xử lý làm sạch text trước khi train/predict.

### `preprocessing/preprocessor.py`

Chức năng:

- normalize unicode
- lowercase
- xóa HTML, URL
- chuẩn hóa teencode
- xóa dấu câu
- xóa chữ số
- word segmentation nếu có `underthesea`
- remove stopwords nếu bật

Class chính:

```python
Preprocessor
```

Hàm quan trọng:

```python
clean(text)
clean_batch(texts)
```

Tất cả model nên dùng cùng preprocessor để đảm bảo train và predict nhất quán.

## 5. algorithms/

Thư mục `algorithms/` chứa class thuật toán. Mỗi thuật toán nên có interface
giống nhau để train/evaluate/predict chung.

Interface nên có:

```python
fit(X_train, y_train)
predict(X)
predict_proba(X)
predict_one(text)
```

Nếu thuật toán có thể giải thích feature, có thể thêm:

```python
top_features(n=15)
```

### `algorithms/naive_bayes.py`

Chức năng:

- cài đặt Naive Bayes cho text classification
- dùng TF-IDF vectorizer
- tính class prior và feature likelihood
- predict bằng joint log likelihood

Phù hợp:

- baseline nhanh
- dễ so sánh với model mạnh hơn

### `algorithms/random_forest.py`

Chức năng:

- wrap `RandomForestClassifier`
- dùng `TfidfVectorizer`
- cho truyền các tham số:

```python
n_estimators
max_depth
ngram_range
max_features
sublinear_tf
min_samples_split
min_samples_leaf
class_weight
random_state
```

- có `top_features()` dựa trên `feature_importances_`

Phù hợp:

- so sánh với model tuyến tính
- thử nghiệm class_weight và oversampling

### `algorithms/logistic_regression.py`

Chức năng:

- wrap `LogisticRegression`
- dùng `TfidfVectorizer`
- cho truyền các tham số cần tune:

```python
C
max_iter
ngram_range
max_features
sublinear_tf
class_weight
random_state
```

- hỗ trợ:

```python
fit(X_train, y_train, sample_weight=None)
```

- có `top_features()` dựa trên trọng số `coef_` theo từng class

Phù hợp:

- text classification
- train nhanh
- kết quả tốt
- xử lý imbalance tốt bằng `class_weight="balanced"`

### `algorithms/svm.py`

Chức năng dự kiến:

- wrap Linear SVM hoặc SVM classifier
- dùng TF-IDF
- cho phép tune `C`, `class_weight`, `ngram_range`, `max_features`

Phù hợp:

- text classification
- baseline mạnh cho bài toán sentiment

### `algorithms/__init__.py`

Export các class thuật toán để import gọn hơn:

```python
from .naive_bayes import NaiveBayesClassifier
from .random_forest import RandomForestSentiment
from .logistic_regression import LogisticRegressionSentiment
from .svm import SVMClassifier
```

## 6. evaluation/

Thư mục `evaluation/` chứa logic đánh giá model.

### `evaluation/evaluator.py`

Chức năng:

- evaluate model trên tập dev/test
- tính metric tổng thể
- tính metric từng class
- in classification report
- in confusion matrix

`evaluate()` trả về:

```python
{
    "name": ...,
    "accuracy": ...,
    "precision": ...,
    "recall": ...,
    "f1": ...,
    "train_time": ...,
    "label_scores": ...,
    "confusion_matrix": ...,
    "y_true": ...,
    "y_pred": ...,
}
```

`label_scores` chứa metric riêng cho từng class:

```python
{
    "label": 1,
    "name": "Neutral",
    "correct": ...,
    "total": ...,
    "percent": ...,
    "precision": ...,
    "recall": ...,
    "f1": ...,
}
```

Metric cần theo dõi:

```text
Accuracy
Precision
Recall
Weighted F1
Macro F1
Neutral F1
Confusion Matrix
```

Với dataset lệch lớp, không nên chỉ nhìn Accuracy.

## 7. experiments/

Thư mục `experiments/` dùng để tune tham số trên dev.

Nguyên tắc:

```text
train.csv -> train
dev.csv   -> tune
test.csv  -> không dùng trong experiments
```

Mỗi thuật toán nên có một file tune riêng.

### `experiments/tune_random_forest.py`

Chức năng:

- thử nhiều config Random Forest
- train trên train set
- evaluate trên dev set
- in bảng:

```text
params | dev Accuracy | dev F1 | Neutral F1
```

- chọn best theo weighted dev F1
- theo dõi best Neutral F1 riêng
- dùng `utils/imbalance.py` để oversample Neutral

Có xử lý imbalance:

```python
class_weight="balanced"
neutral_multiplier=3/5/8
```

### `experiments/tune_logistic_regression.py`

Chức năng:

- tune Logistic Regression trên dev
- thử các giá trị:

```python
C
ngram_range
max_features
sublinear_tf
class_weight
```

- chọn best theo weighted dev F1
- theo dõi Neutral F1 riêng

Best config hiện tại:

```python
{
    "C": 2.0,
    "max_iter": 1000,
    "ngram_range": (1, 2),
    "max_features": 30000,
    "sublinear_tf": True,
    "class_weight": "balanced",
}
```

### Tune cho thuật toán khác

Nếu thêm SVM, Decision Tree, KNN, nên tạo:

```text
experiments/tune_svm.py
experiments/tune_decision_tree.py
experiments/tune_knn.py
```

Mỗi file tune nên có:

```python
PARAM_GRID = [...]
get_label_f1(...)
print_row(...)
main()
```

## 8. training/

Thư mục `training/` dùng để train final model.

Khác với `experiments/`:

```text
experiments/ -> tune trên dev
training/    -> train final và evaluate trên test
```

### `training/train_random_forest.py`

Chức năng:

- load train/test
- preprocess
- train Random Forest bằng best params
- evaluate trên test
- in top features
- save model vào `models/random_forest.joblib`

### `training/train_logistic_regression.py`

Chức năng:

- load train/test
- preprocess
- train Logistic Regression bằng best params
- evaluate trên test
- in top features
- save model vào `models/logistic_regression.joblib`

### `training/train_naive_bayes.py`

Chức năng:

- train baseline Naive Bayes
- evaluate trên test
- save model vào `models/naive_bayes.joblib`

### File train cho thuật toán mới

Nếu thêm SVM:

```text
training/train_svm.py
```

Flow:

```text
load train/test
preprocess
train final bằng best params
evaluate test
save model
```

## 9. utils/

Thư mục `utils/` chứa helper dùng chung.

### `utils/model_manager.py`

Chức năng:

- save model bằng joblib
- load model
- kiểm tra model có tồn tại không

Hàm:

```python
save_model(model, path)
load_model(path)
model_exists(path)
```

### `utils/imbalance.py`

Chức năng:

- xử lý mất cân bằng lớp ở tầng dữ liệu
- dùng chung cho nhiều thuật toán

Hàm:

```python
label_distribution(y)
print_label_distribution(y)
oversample_label(X, y, label=1, multiplier=3)
oversample_neutral(X, y, multiplier=3)
oversample_to_max_class(X, y)
```

Khi nào dùng:

- lớp Neutral quá ít
- muốn tăng Neutral F1
- muốn so sánh oversampling với class_weight

Lưu ý:

- oversampling có thể tăng Neutral F1
- nhưng có thể làm giảm weighted F1 hoặc Accuracy
- phải so sánh trên dev, không tune trên test

## 10. prediction/

Thư mục `prediction/` dùng để predict/demo model đã save.

### `prediction/predict.py`

Chức năng:

- load các model có sẵn trong `models/`
- preprocess câu người dùng nhập
- predict bằng tất cả model hiện có
- model nào chưa có file `.joblib` thì tự bỏ qua

## 11. models/

Thư mục `models/` chứa model đã train và save bằng joblib.

Ví dụ:

```text
models/naive_bayes.joblib
models/random_forest.joblib
models/logistic_regression.joblib
models/svm.joblib
```

Lưu ý:

- file `.joblib` là binary
- có thể nặng
- không bắt buộc push lên GitHub
- nên để `models/` trong `.gitignore`

## 12. docs/

Thư mục `docs/` chứa tài liệu project.

Tài liệu nên có:

```text
algorithm_upgrade_flow.md
project_flow_and_files.md
```

Mục đích:

- ghi lại flow nâng cấp thuật toán
- giải thích chức năng file
- giúp người khác đọc project nhanh hơn

## 13. main.py

`main.py` có thể dùng làm script demo tổng quát hoặc entrypoint cũ.

Nếu project đã có các file riêng:

```text
training/
experiments/
prediction/
```

thì `main.py` chỉ nên giữ vai trò demo đơn giản, không nên chứa quá nhiều logic.


## 14. Xử lý mất cân bằng lớp

UIT-VSFC bị lệch lớp:

```text
Neutral ít hơn Negative/Positive rất nhiều
```

Vì vậy cần theo dõi:

```text
Neutral Recall
Neutral F1
Macro F1
```

Hướng xử lý:

```text
class_weight="balanced"
oversampling Neutral
ComplementNB
sample_weight
tune theo macro F1
tune theo Neutral F1
```

Phần dùng chung:

```python
from utils.imbalance import oversample_label, NEUTRAL_LABEL
```

Phần phụ thuộc model:

```python
class_weight="balanced"
```

Không phải thuật toán nào cũng hỗ trợ `class_weight`.

