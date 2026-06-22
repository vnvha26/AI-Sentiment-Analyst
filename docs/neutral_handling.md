# Xu ly lop Neutral

## 1. Van de chung

Trong bo du lieu UIT-VSFC, lop `Neutral` co so mau it hon nhieu so voi hai lop `Negative` va `Positive`.

Phan bo du lieu hien tai:

| Tap du lieu | Negative | Neutral | Positive | Tong |
|---|---:|---:|---:|---:|
| Train | 5325 | 458 | 5643 | 11426 |
| Dev | 705 | 73 | 805 | 1583 |
| Test | 1409 | 167 | 1590 | 3166 |

Do lop `Neutral` chi chiem ti le nho, model co xu huong uu tien du doan `Negative` hoac `Positive` de dat Accuracy/F1 tong the cao. Vi vay can theo doi rieng:

- `Macro F1`
- `Neutral F1`
- Precision/Recall/F1 cua tung lop

Khong nen chi nhin `Accuracy` hoac `weighted F1`, vi hai chi so nay co the che lap diem yeu cua lop `Neutral`.

---

## 2. Naive Bayes

### 2.1. Diem yeu voi lop Neutral

Naive Bayes tinh xac suat dua tren tan suat tu/cum tu xuat hien theo tung nhan. Voi lop `Neutral`, so mau train it nen:

- xac suat cua cac tu thuoc lop `Neutral` khong on dinh;
- model de bi nghieng ve `Negative`/`Positive`;
- nhieu cau trung tinh bi doan thanh cam xuc ro hon;
- neu khong xu ly mat can bang, `Neutral F1` co the rat thap hoac bang `0%`.

Vi du trong qua trinh tune, nhieu config khong xu ly can bang cho ket qua:

```text
Neutral F1 = 0.00%
```

Dieu nay cho thay model gan nhu khong nhan dien duoc lop `Neutral`.

---

### 2.2. Cac cach da thu

#### Cach 1: Oversampling Neutral

Oversampling la nhan them mau cua lop `Neutral` trong tap train.

Tham so:

```text
neutral_multiplier
```

Y nghia:

```text
neutral_multiplier = 1  -> khong nhan them Neutral
neutral_multiplier = 3  -> moi mau Neutral duoc lap thanh 3 mau
```

Ket qua tot nhat truoc khi them sample weight:

```text
neutral_multiplier = 3
neutral_weight     = 1

Dev weighted F1  = 87.89%
Dev macro F1     = 69.50%
Dev Neutral F1   = 26.97%
```

Nhan xet:

- Tot hon so voi khong xu ly mat can bang.
- Nhung cach nay lap lai du lieu Neutral, co the lam model hoc lap mau cu.

---

#### Cach 2: Sample weight cho Neutral

Sample weight khong nhan ban du lieu. Thay vao do, moi mau `Neutral` duoc gan trong so lon hon khi model hoc.

Tham so:

```text
neutral_weight
```

Y nghia:

```text
neutral_weight = 1  -> Neutral co trong so nhu cac lop khac
neutral_weight = 5  -> loi/su dong gop cua Neutral duoc tinh nang hon 5 lan
```

Config tot nhat hien tai:

```json
{
  "alpha": 0.3,
  "ngram_range": [1, 2],
  "max_features": 30000,
  "sublinear_tf": true,
  "fit_prior": true,
  "class_prior": null,
  "neutral_multiplier": 1,
  "neutral_weight": 5
}
```

Ket qua tren dev:

```text
Accuracy       = 89.64%
Weighted F1    = 89.00%
Macro F1       = 74.10%
Neutral F1     = 39.62%
```

Ket qua tren test:

```text
Accuracy       = 87.30%
Weighted F1    = 86.22%
Macro F1       = 67.00%
Neutral F1     = 21.55%
```

Nhan xet:

- `Neutral F1` tren dev tang tu `26.97%` len `39.62%`.
- `Neutral F1` tren test tang tu `15.62%` len `21.55%`.
- So cau Neutral du doan dung tren test tang tu `15/167` len `25/167`.
- Accuracy giam nhe tu `87.71%` xuong `87.30%`, nhung weighted F1 tang tu `86.07%` len `86.22%`.
- Day la mot danh doi hop ly: model nhan dien Neutral tot hon trong khi hieu nang tong the gan nhu duoc giu nguyen.

---

### 2.3. Cach dang dung hien tai

Naive Bayes hien tai dang dung:

```text
alpha              = 0.3
neutral_multiplier = 1
neutral_weight     = 5
```

Tuc la:

- khong oversampling Neutral;
- khong nhan ban mau Neutral;
- chi tang trong so cua mau Neutral khi train.

File lien quan:

```text
utils/imbalance.py
experiments/tune_naive_bayes.py
training/train_naive_bayes.py
configs/naive_bayes_best.json
```

Trong `utils/imbalance.py`, ham tao sample weight:

```python
sample_weight_for_label(y, label=NEUTRAL_LABEL, weight=5)
```

Trong `training/train_naive_bayes.py`, model duoc train voi:

```python
model.fit(X_fit, y_fit, sample_weight=sample_weight)
```

---

### 2.4. Ket luan cho Naive Bayes

Voi Naive Bayes, cach xu ly `Neutral` tot nhat hien tai la dung `sample_weight`, khong dung oversampling.

Ly do:

- sample weight giup lop `Neutral` co anh huong lon hon khi hoc;
- khong lam tang kich thuoc tap train;
- giam viec model hoc lap lai cung mot mau Neutral;
- ket qua dev cho thay `Macro F1` va `Neutral F1` tang ro;
- ket qua test xac nhan `Neutral F1` tang tu `15.62%` len `21.55%`.

Tuy nhien, lop `Neutral` van la diem yeu cua Naive Bayes do so mau it va cau trung tinh thuong mo ho. Neu muon cai thien tiep, co the thu:

- thu `fit_prior=False` ket hop voi sample weight;
- thu Complement Naive Bayes.

---

## 3. Logistic Regression

### 3.1. Diem yeu voi lop Neutral

Logistic Regression phu hop voi vector TF-IDF, nhung du lieu bi lech lop lam ranh gioi phan loai nghieng ve `Negative` va `Positive`.

Khi khong tang trong so cho Neutral, model co the dat Accuracy va weighted F1 cao nhung bo sot nhieu cau Neutral. Vi vay qua trinh tune theo doi them:

- `Macro F1`;
- `Neutral F1`;
- Precision va Recall rieng cua Neutral.

---

### 3.2. Cac cach da thu

#### Cach 1: Class weight balanced

Tune da thu:

```python
class_weight="balanced"
```

`class_weight="balanced"` tu dong tang trong so cua lop it mau. Cach nay giup Neutral tot hon, nhung co the lam Accuracy va weighted F1 giam.

Ket qua dev tot nhat trong nhom balanced:

```text
C               = 2.0
class_weight    = balanced
neutral_weight  = 1

Accuracy        = 89.51%
Weighted F1     = 89.68%
Macro F1        = 75.06%
Neutral F1      = 41.29%
```

---

#### Cach 2: Sample weight cho Neutral

Thay vi de sklearn tu can bang tat ca cac lop, project tang rieng trong so cho mau Neutral:

```python
sample_weight_for_label(y, label=NEUTRAL_LABEL, weight=5)
```

Config duoc chon theo weighted dev F1:

```json
{
  "C": 1.0,
  "max_iter": 1000,
  "ngram_range": [1, 2],
  "max_features": 30000,
  "sublinear_tf": true,
  "class_weight": null,
  "neutral_weight": 5
}
```

Ket qua tren dev:

```text
Accuracy        = 90.97%
Weighted F1     = 90.66%
Macro F1        = 76.32%
Neutral F1      = 43.20%
```

Ket qua tren test:

```text
Accuracy        = 88.57%
Weighted F1     = 88.02%
Macro F1        = 72.00%
Neutral F1      = 33.09%
Neutral dung    = 45/167
```

Chi tiet Neutral tren test:

```text
Precision       = 42.86%
Recall          = 26.95%
F1              = 33.09%
```

---

### 3.3. Cach dang dung hien tai

Logistic Regression hien tai dung:

```text
C              = 1.0
class_weight   = None
neutral_weight = 5
```

Tuc la:

- khong dung `class_weight="balanced"`;
- khong nhan ban du lieu Neutral;
- gan trong so `5` cho moi mau Neutral khi train;
- van chon config chinh theo weighted dev F1;
- theo doi Macro F1 va Neutral F1 de kiem tra mat can bang lop.

File lien quan:

```text
utils/imbalance.py
experiments/tune_logistic_regression.py
training/train_logistic_regression.py
configs/logistic_regression_best.json
```

Trong `training/train_logistic_regression.py`, model duoc train voi:

```python
model.fit(X_train_clean, y_train, sample_weight=sample_weight)
```

---

### 3.4. Ket luan cho Logistic Regression

Voi Logistic Regression, `sample_weight=5` cho ket qua can bang hon `class_weight="balanced"` trong grid hien tai:

- weighted dev F1 cao hon;
- Macro F1 cao hon;
- Neutral F1 tren dev cao hon;
- ket qua test giu duoc Accuracy `88.57%` va Neutral F1 `33.09%`.

Neutral van la lop kho nhat do so mau it, nhung cach dung sample weight da giup model nhan dung `45/167` cau Neutral tren tap test.
