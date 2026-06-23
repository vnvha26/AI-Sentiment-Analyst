# Phân loại cảm xúc tiếng Việt với UIT-VSFC

Project phân loại phản hồi sinh viên tiếng Việt thành ba nhãn cảm xúc bằng các thuật toán Machine Learning truyền thống và đặc trưng TF-IDF.

Các thuật toán được sử dụng:

- Naive Bayes
- Logistic Regression
- Random Forest
- SVM (`LinearSVC`)

Pipeline chính:

```text
Đọc dữ liệu
-> tiền xử lý tiếng Việt
-> TF-IDF
-> tune tham số trên dev
-> train model final
-> đánh giá trên test
-> lưu model
-> dự đoán câu mới
```

## 1. Yêu cầu môi trường

Khuyến nghị:

- Python 3.10 hoặc 3.11
- Git
- Windows PowerShell, Command Prompt hoặc Git Bash

Các thư viện chính:

```text
numpy
pandas
scikit-learn
joblib
underthesea
streamlit
```

`underthesea` được dùng để ghép từ tiếng Việt. Nếu không cài, project vẫn chạy nhưng sẽ bỏ qua bước ghép từ.

## 2. Tải project

```bash
git clone https://github.com/vnvha26/AI-Sentiment-Analyst.git
cd AI-Sentiment-Analyst
```

Nếu đã có project ở máy thì chỉ cần mở terminal tại thư mục gốc, nơi chứa `README.md`, `algorithms/`, `training/` và `experiments/`.

## 3. Tạo môi trường Python

### Dùng venv trên Windows

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn joblib underthesea streamlit
```

## 4. Chuẩn bị dữ liệu

Project sử dụng UIT-VSFC với ba tập:

| Tập | Mục đích |
|---|---|
| `train` | Huấn luyện model và fit TF-IDF |
| `dev` | Tune tham số và chọn best config |
| `test` | Đánh giá cuối cùng sau khi chọn config |

Các nhãn cảm xúc:

```text
0: Negative
1: Neutral
2: Positive
```

Nếu chưa có dữ liệu, chạy:

```bash
python data/loader.py
```

Lệnh trên tải dữ liệu raw và tạo:

```text
data/uit-vsfc-sentiment/train.csv
data/uit-vsfc-sentiment/dev.csv
data/uit-vsfc-sentiment/test.csv
```

Nếu các file CSV đã tồn tại thì có thể bỏ qua bước này.

## 5. Cấu trúc chính

```text
algorithms/       Cài đặt/wrapper các thuật toán
configs/          Best params được tạo sau khi tune
data/             Loader và dữ liệu UIT-VSFC
evaluation/       Accuracy, Precision, Recall, F1, Confusion Matrix
experiments/      Tune tham số trên train/dev
models/           Model .joblib được tạo sau khi train
prediction/       Dự đoán bằng các model đã lưu
preprocessing/    Làm sạch, teencode, ghép từ và phủ định
training/         Train final, test và lưu model
utils/            Model manager và xử lý mất cân bằng lớp
app.py            Giao diện Streamlit
```

## 6. Tiền xử lý dữ liệu

File chính:

```text
preprocessing/preprocessor.py
```

Các bước xử lý gồm:

- chuẩn hóa Unicode;
- chuyển chữ thường;
- loại HTML và URL;
- chuẩn hóa token đặc biệt trong UIT-VSFC;
- chuẩn hóa teencode;
- loại dấu câu và chữ số;
- ghép từ tiếng Việt bằng `underthesea`;
- xử lý cụm phủ định.

Logic ghép phủ định nằm tại:

```text
preprocessing/negation.py
```

Ví dụ:

```text
không vui      -> không_vui
chưa tốt       -> chưa_tốt
không dễ hiểu  -> không_dễ_hiểu
```

Nếu thay đổi logic preprocessing, cần tune và train lại các model vì không gian đặc trưng TF-IDF đã thay đổi.

## 7. Tune tham số

Tune luôn dùng:

```text
train -> fit model
dev   -> đánh giá và chọn config
test  -> chưa sử dụng
```

### Naive Bayes

```bash
python experiments/tune_naive_bayes.py
```

Best params được lưu vào:

```text
configs/naive_bayes_best.json
```

### Logistic Regression

```bash
python experiments/tune_logistic_regression.py
```

Best params được lưu vào:

```text
configs/logistic_regression_best.json
```

### Random Forest

```bash
python experiments/tune_random_forest.py
```

Best params được lưu vào:

```text
configs/random_forest_best.json
```

### SVM

```bash
python experiments/tune_svm.py
```

Best params được lưu vào:

```text
configs/svm_best.json
```

SVM thử nhiều cấu hình hơn nên có thể chạy lâu hơn ba model còn lại.

## 8. Train model final

Sau khi tune, chạy các file train để đọc best config, đánh giá trên test và lưu model.

### Naive Bayes

```bash
python training/train_naive_bayes.py
```

Model được lưu tại:

```text
models/naive_bayes.joblib
```

### Logistic Regression

```bash
python training/train_logistic_regression.py
```

Model được lưu tại:

```text
models/logistic_regression.joblib
```

### Random Forest

```bash
python training/train_random_forest.py
```

Model được lưu tại:

```text
models/random_forest.joblib
```

### SVM

```bash
python training/train_svm_final.py
```

Model được lưu tại:

```text
models/svm_best.joblib
```

Thư mục `models/` và file `.joblib` đang được bỏ qua bởi `.gitignore`. Khi clone project trên máy mới, cần train lại model hoặc tự chuyển các file model đã train vào thư mục này.

## 9. Chạy toàn bộ quy trình

Thứ tự đầy đủ:

```bash
python data/loader.py

python experiments/tune_naive_bayes.py
python experiments/tune_logistic_regression.py
python experiments/tune_random_forest.py
python experiments/tune_svm.py

python training/train_naive_bayes.py
python training/train_logistic_regression.py
python training/train_random_forest.py
python training/train_svm_final.py
```

Không cần chạy lại tune mỗi lần dự đoán. Chỉ cần tune/train lại khi:

- thay đổi preprocessing;
- thay đổi grid tham số;
- thay đổi dữ liệu;
- thay đổi logic thuật toán;
- chưa có model `.joblib`.

## 10. Dự đoán trên terminal

Chạy tất cả model hiện có trong `models/`:

```bash
python prediction/predict.py
```

Nhập câu rồi nhấn Enter. Nhập `quit`, `exit`, `q` hoặc để trống để thoát.

Dự đoán trực tiếp một câu:

```bash
python prediction/predict.py --text "hôm nay không vui"
```

Chỉ dùng một model:

```bash
python prediction/predict.py --model logistic_regression --text "môn học khá bình thường"
```

Các lựa chọn cho `--model`:

```text
all
naive_bayes
logistic_regression
random_forest
svm
```

File model nào chưa tồn tại sẽ được bỏ qua khi chọn `all`.

## 11. Chạy giao diện Streamlit

Đảm bảo đã train ít nhất một model, sau đó chạy:

```bash
streamlit run app.py
```

Nếu lệnh `streamlit` không được nhận diện:

```bash
python -m streamlit run app.py
```

Mở trình duyệt tại:

```text
http://localhost:8501
```

Giao diện gồm:

- thông tin train/dev/test;
- phân bố nhãn;
- kết quả các model trên test;
- nhận xét ưu/nhược điểm;
- dự đoán cảm xúc câu mới bằng các model đã lưu.

## 12. Đánh giá model

Project báo cáo:

- Accuracy;
- Precision;
- Recall;
- weighted F1;
- F1 từng lớp;
- Macro F1;
- Confusion Matrix;
- số câu dự đoán đúng trên từng lớp.

Do lớp `Neutral` ít mẫu, không nên chỉ nhìn Accuracy. Cần theo dõi thêm:

```text
Macro F1
Neutral Precision
Neutral Recall
Neutral F1
```

