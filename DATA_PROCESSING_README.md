# Xử lý dữ liệu

Project có 2 file xử lý dữ liệu chính:

```text
data/loader.py
preprocessing/preprocessor.py
```

## 1. `data/loader.py`

File này xử lý dữ liệu ở mức **dataset/file**.

Dataset UIT-VSFC có:

```text
sentences   -> câu phản hồi
sentiments  -> nhãn cảm xúc
topics      -> nhãn chủ đề
```

Project này chỉ làm Sentiment Analysis nên chỉ dùng:

```text
sentences + sentiments
```

Không dùng `topics`.

File này sẽ:

1. Tải raw `sentences` và `sentiments`.
2. Ghép từng câu với nhãn cảm xúc tương ứng.
3. Tạo CSV:

```text
data/uit-vsfc-sentiment/train.csv
data/uit-vsfc-sentiment/dev.csv
data/uit-vsfc-sentiment/test.csv
```

CSV có dạng:

```csv
sentence,sentiment
thầy giảng rất dễ hiểu,2
môn học quá khó,0
```

Ý nghĩa nhãn:

```text
0 = Negative
1 = Neutral
2 = Positive
```

## 2. `preprocessing/preprocessor.py`

File này xử lý ở mức **từng câu văn bản** trước khi đưa vào model.

Input:

```text
Thầy giảng rất dễ hiểu!!!
```

Output ví dụ:

```text
thầy giảng rất dễ_hiểu
```

Các bước chính:

1. Chuẩn hóa Unicode tiếng Việt.
2. Chuyển về chữ thường.
3. Xóa HTML/URL.
4. Chuẩn hóa teencode.

```text
ko -> không
đc -> được
bt -> bình thường
sv -> sinh viên
```

5. Xóa dấu câu và chữ số.
6. Xóa khoảng trắng thừa.
7. Ghép từ tiếng Việt bằng `underthesea` nếu có cài:

```text
sinh viên -> sinh_viên
dễ hiểu -> dễ_hiểu
```

8. Có thể xóa stopwords nếu bật, nhưng mặc định không xóa vì các từ như `không`, `chưa`, `rất`, `quá` rất quan trọng cho cảm xúc.

## 3. Thứ tự pipeline

```text
loader
-> preprocessor
-> vectorizer
-> model
```

Nói ngắn gọn:

- `loader.py`: tạo dữ liệu CSV từ raw.
- `preprocessor.py`: làm sạch câu trước khi train/predict.
