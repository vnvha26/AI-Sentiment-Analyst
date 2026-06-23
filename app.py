import contextlib
import io
import os
import sys
from collections import Counter

import pandas as pd
import streamlit as st


PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

from data.loader import LABEL_MAP, load_dataset
from evaluation.evaluator import evaluate
from preprocessing.preprocessor import Preprocessor
from utils.model_manager import load_model


DATA_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "uit-vsfc-sentiment"),
    os.path.join(PROJECT_ROOT, "data", "data", "uit-vsfc-sentiment"),
]

MODEL_PATHS = {
    "Naive Bayes": os.path.join(PROJECT_ROOT, "models", "naive_bayes.joblib"),
    "Logistic Regression": os.path.join(
        PROJECT_ROOT,
        "models",
        "logistic_regression.joblib",
    ),
    "Random Forest": os.path.join(PROJECT_ROOT, "models", "random_forest.joblib"),
    "SVM": os.path.join(PROJECT_ROOT, "models", "svm_best.joblib"),
}

LABEL_COLORS = {
    0: "#dc2626",
    1: "#ca8a04",
    2: "#16a34a",
}

EXAMPLES = [
    "Môn học có điểm danh đầy đủ, tài liệu được gửi trên hệ thống theo từng tuần.",
    "Hôm nay học không vui, bài giảng hơi khó theo kịp.",
    "Hôm nay đi học rất bình thường",
    "Giảng viên giảng khó hiểu, ít ví dụ và phản hồi sinh viên khá chậm.",
    "Tối nay quán phở gần nhà nấu rất ngon và phục vụ nhanh.",
]

MODEL_NOTES = [
    {
        "Model": "Naive Bayes",
        "Ưu điểm": "Train rất nhanh, đơn giản, phù hợp làm baseline.",
        "Nhược điểm": "Yếu với Neutral và các ngữ cảnh phủ định phức tạp.",
    },
    {
        "Model": "Logistic Regression",
        "Ưu điểm": "F1 ổn định, phù hợp với đặc trưng TF-IDF của văn bản.",
        "Nhược điểm": "Vẫn có thể nhầm Neutral khi câu không có cảm xúc rõ.",
    },
    {
        "Model": "Random Forest",
        "Ưu điểm": "Có feature importance, chạy được với pipeline hiện tại.",
        "Nhược điểm": "Kém phù hợp hơn model tuyến tính với vector text thưa nhiều chiều.",
    },
    {
        "Model": "SVM",
        "Ưu điểm": "Accuracy và weighted F1 cao với bài toán phân loại văn bản.",
        "Nhược điểm": "Neutral F1 còn hạn chế, flow chưa đồng bộ bằng các model khác.",
    },
]


def apply_global_styles():
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-size: 18px;
        }
        .block-container {
            padding-top: 2rem;
            max-width: 1320px;
        }
        h1 {
            font-size: 2.7rem !important;
        }
        h2, h3 {
            font-size: 1.65rem !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
        }
        div[data-testid="stDataFrame"] {
            font-size: 1.05rem;
        }
        .stTextArea textarea,
        .stSelectbox div,
        .stButton button {
            font-size: 1.05rem !important;
        }
        section[data-testid="stSidebar"] {
            font-size: 1.05rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_data_dir():
    for data_dir in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(data_dir, "train.csv")):
            return data_dir
    return DATA_DIR_CANDIDATES[0]


@st.cache_resource
def get_preprocessor():
    return Preprocessor()


@st.cache_resource
def load_available_models():
    models = {}
    missing = {}

    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[name] = load_model(path)
        else:
            missing[name] = path

    return models, missing


@st.cache_data
def load_dataset_silent(data_dir):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return load_dataset(data_dir)


@st.cache_data(show_spinner=False)
def clean_texts_silent(texts):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return Preprocessor().clean_batch(texts)


def build_dataset_summary(splits):
    rows = []

    for split, (_, labels) in splits.items():
        counts = Counter(labels)
        total = len(labels)

        for label in sorted(LABEL_MAP):
            count = counts.get(label, 0)
            rows.append({
                "Split": split,
                "Label": LABEL_MAP[label],
                "Count": count,
                "Percent": count / total * 100 if total else 0,
            })

    return pd.DataFrame(rows)


def build_split_usage(splits):
    return pd.DataFrame([
        {
            "Split": "train",
            "Số mẫu": len(splits["train"][0]),
            "Mục đích": "Dùng để model học.",
            "Dùng trong": "training/train_*.py, experiments/tune_*.py",
        },
        {
            "Split": "dev",
            "Số mẫu": len(splits["dev"][0]),
            "Mục đích": "Dùng để chọn cấu hình/tham số tốt nhất.",
            "Dùng trong": "experiments/tune_*.py",
        },
        {
            "Split": "test",
            "Số mẫu": len(splits["test"][0]),
            "Mục đích": "Dùng để đánh giá cuối sau khi đã chọn cấu hình.",
            "Dùng trong": "training/train_*.py",
        },
    ])


def build_label_count_table(summary):
    table = summary.pivot(index="Split", columns="Label", values="Count")
    return table.reindex(index=["train", "dev", "test"])


def get_label_f1(result, label_id):
    for item in result.get("label_scores", []):
        if item["label"] == label_id:
            return item.get("f1", 0)
    return 0


def evaluate_saved_models(models, splits):
    X_test, y_test = splits["test"]
    X_test_clean = clean_texts_silent(X_test)
    rows = []

    for model_name, model in models.items():
        result = evaluate(model, X_test_clean, y_test)
        rows.append({
            "Model": model_name,
            "Accuracy": result["accuracy"] * 100,
            "Precision": result["precision"] * 100,
            "Recall": result["recall"] * 100,
            "F1": result["f1"] * 100,
            "Neutral F1": get_label_f1(result, 1) * 100,
            "Train(s)": result["train_time"],
        })

    return pd.DataFrame(rows)


def format_result_table(df):
    if df.empty:
        return df

    formatted = df.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1", "Neutral F1"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2f}%")
    formatted["Train(s)"] = formatted["Train(s)"].map(lambda value: f"{value:.3f}")
    return formatted


def predict_with_models(models, preprocessor, text):
    cleaned = preprocessor.clean(text)
    predictions = []

    for model_name, model in models.items():
        if hasattr(model, "predict_one"):
            label, confidence = model.predict_one(cleaned)
        else:
            label = int(model.predict([cleaned])[0])
            confidence = None

        predictions.append({
            "Model": model_name,
            "Prediction": LABEL_MAP.get(int(label), str(label)),
            "Label": int(label),
            "Confidence": confidence,
        })

    return cleaned, predictions


def majority_vote(predictions):
    if not predictions:
        return None

    votes = Counter(item["Label"] for item in predictions)
    max_votes = max(votes.values())
    candidates = {label for label, count in votes.items() if count == max_votes}

    if len(candidates) == 1:
        label = next(iter(candidates))
        return label, max_votes, False

    confidence_sum = Counter()
    for item in predictions:
        confidence_sum[item["Label"]] += item["Confidence"] or 0

    label = max(candidates, key=lambda item: confidence_sum[item])
    return label, max_votes, True


def format_confidence(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def render_prediction_card(label, vote_count, is_tie):
    label_name = LABEL_MAP.get(label, str(label))
    color = LABEL_COLORS.get(label, "#334155")
    tie_text = "Hòa phiếu, chọn theo confidence" if is_tie else "Bỏ phiếu đa số"

    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {color};
            padding: 14px 16px;
            background: #f8fafc;
            border-radius: 6px;
            margin: 8px 0 18px 0;
        ">
            <div style="font-size: 16px; color: #475569;">Kết quả tổng hợp</div>
            <div style="font-size: 28px; font-weight: 700; color: {color};">
                {label_name} ({label})
            </div>
            <div style="font-size: 16px; color: #64748b;">
                {tie_text}, số phiếu: {vote_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_status(models, missing):
    rows = []
    for name in MODEL_PATHS:
        rows.append({
            "Model": name,
            "Trạng thái": "Có model" if name in models else "Thiếu model",
            "Đường dẫn": MODEL_PATHS[name],
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if missing:
        st.caption("Model bị thiếu sẽ được bỏ qua khi dự đoán.")


def render_overview_tab(models):
    st.subheader("Tổng quan dữ liệu và kết quả model")

    data_dir = find_data_dir()
    st.caption(f"Thư mục dữ liệu: {data_dir}")

    try:
        splits = load_dataset_silent(data_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    summary = build_dataset_summary(splits)
    total_samples = sum(len(texts) for texts, _ in splits.values())

    col_train, col_dev, col_test, col_total = st.columns(4)
    col_train.metric("Train", len(splits["train"][0]))
    col_dev.metric("Dev", len(splits["dev"][0]))
    col_test.metric("Test", len(splits["test"][0]))
    col_total.metric("Total", total_samples)

    st.markdown("**Cách sử dụng các tập dữ liệu**")
    st.dataframe(build_split_usage(splits), use_container_width=True, hide_index=True)

    st.markdown("**Số lượng câu theo từng nhãn**")
    label_counts = build_label_count_table(summary)
    st.dataframe(label_counts, use_container_width=True)

    st.markdown("**Kết quả của từng model trên tập test**")
    if models:
        with st.spinner("Đang đánh giá các model đã lưu trên tập test..."):
            results = evaluate_saved_models(models, splits)
        st.dataframe(format_result_table(results), use_container_width=True, hide_index=True)
    else:
        st.warning("Chưa tìm thấy model đã lưu trong thư mục models/.")

    st.markdown("**Nhận xét ưu/nhược điểm của từng thuật toán**")
    st.dataframe(pd.DataFrame(MODEL_NOTES), use_container_width=True, hide_index=True)


def render_predict_tab(models, preprocessor):
    st.subheader("Dự đoán cảm xúc")

    if "input_text" not in st.session_state:
        st.session_state.input_text = EXAMPLES[0]

    selected_example = st.selectbox("Câu ví dụ", EXAMPLES)
    if st.button("Dùng câu ví dụ"):
        st.session_state.input_text = selected_example

    text = st.text_area(
        "Câu đầu vào",
        key="input_text",
        height=110,
    )

    if not models:
        st.error("Chưa tìm thấy model trong thư mục models/. Hãy train ít nhất một model.")
        return

    if st.button("Dự đoán", type="primary"):
        text = text.strip()
        if not text:
            st.warning("Hãy nhập một câu trước khi dự đoán.")
            return

        cleaned, predictions = predict_with_models(models, preprocessor, text)
        vote = majority_vote(predictions)

        if vote is not None:
            render_prediction_card(*vote)

        st.markdown("**Tiền xử lý**")
        st.code(f"Input  : {text}\nCleaned: {cleaned}", language="text")

        result_rows = [
            {
                "Model": item["Model"],
                "Dự đoán": f"{item['Prediction']} ({item['Label']})",
                "Confidence": format_confidence(item["Confidence"]),
            }
            for item in predictions
        ]
        st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Phân loại cảm xúc",
        page_icon=None,
        layout="wide",
    )
    apply_global_styles()

    st.title("Phân loại cảm xúc")
    st.caption("Pipeline TF-IDF với các model Machine Learning truyền thống đã được lưu")

    models, missing = load_available_models()
    preprocessor = get_preprocessor()

    with st.sidebar:
        st.header("Trạng thái model")
        render_model_status(models, missing)

        st.header("Flow project")
        st.markdown(
            """
            `tune_*.py`: train/dev  
            `train_*.py`: train/test  
            `prediction`: model đã lưu
            """
        )

    tab_overview, tab_predict = st.tabs(["Tổng quan", "Dự đoán"])
    with tab_overview:
        render_overview_tab(models)
    with tab_predict:
        render_predict_tab(models, preprocessor)


if __name__ == "__main__":
    main()
