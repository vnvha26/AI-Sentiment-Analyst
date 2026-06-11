import csv
import os
import urllib.request


LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_MAP_VI = {0: "Tieu cuc", 1: "Trung lap", 2: "Tich cuc"}


# Google Drive IDs cua UIT-VSFC raw, chi giu sentences va sentiments.
RAW_DRIVE_IDS = {
    "train": {
        "sentences": "1nzak5OkrheRV1ltOGCXkT671bmjODLhP",
        "sentiments": "1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv",
    },
    "dev": {
        "sentences": "1sMJSR3oRfPc3fe1gK-V3W5F24tov_517",
        "sentiments": "1GiY1AOp41dLXIIkgES4422AuDwmbUseL",
    },
    "test": {
        "sentences": "1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n",
        "sentiments": "1vkQS5gI0is4ACU58-AbWusnemw7KZNfO",
    },
}


def _download_drive(file_id, timeout=60):
    """Tai noi dung 1 file public tren Google Drive bang file ID."""
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def download_raw(raw_dir="data/uit-vsfc-raw-sentiment"):
    """Tai raw sentences/sentiments ve thu muc raw_dir."""
    os.makedirs(raw_dir, exist_ok=True)
    print("Dang tai UIT-VSFC raw cho Sentiment Analysis...")

    for split, files in RAW_DRIVE_IDS.items():
        for kind, file_id in files.items():
            content = _download_drive(file_id)
            out_path = os.path.join(raw_dir, f"{split}_{kind}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
        print(f"  [{split}] da tai sentences/sentiments")

    print(f"Da luu raw vao {raw_dir}/")


def build_csv_from_raw(raw_dir="data/uit-vsfc-raw-sentiment",
                       out_dir="data/uit-vsfc-sentiment"):
    """
    Ghep sentences + sentiments thanh CSV.
    Output gom 3 file: train.csv, dev.csv, test.csv.
    """
    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "dev", "test"]:
        sentence_path = os.path.join(raw_dir, f"{split}_sentences.txt")
        sentiment_path = os.path.join(raw_dir, f"{split}_sentiments.txt")

        if not os.path.exists(sentence_path) or not os.path.exists(sentiment_path):
            raise FileNotFoundError(
                f"Thieu raw file cua split '{split}'. Hay chay download_raw() truoc."
            )

        with open(sentence_path, encoding="utf-8") as f:
            sentences = f.read().splitlines()
        with open(sentiment_path, encoding="utf-8") as f:
            sentiments = f.read().splitlines()

        out_path = os.path.join(out_dir, f"{split}.csv")
        row_count = 0

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sentence", "sentiment"])

            for sentence, sentiment in zip(sentences, sentiments):
                sentence = sentence.strip()
                sentiment = sentiment.strip()
                if sentence and sentiment.isdigit():
                    writer.writerow([sentence, int(sentiment)])
                    row_count += 1

        print(f"  [{split}] {row_count} mau -> {out_path}")

    print(f"Build CSV xong tai {out_dir}/")


def load_csv(filepath):
    """Doc 1 file CSV, tra ve (sentences, labels)."""
    sentences, labels = [], []

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row.get("sentence", "").strip()
            sentiment = row.get("sentiment", "").strip()
            if sentence and sentiment.isdigit():
                sentences.append(sentence)
                labels.append(int(sentiment))

    return sentences, labels


def load_dataset(data_dir="data/uit-vsfc-sentiment"):
    """
    Load train/dev/test tu CSV.
    Neu chua co CSV nhung da co raw thi tu build CSV.
    """
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        raw_dir = "data/uit-vsfc-raw-sentiment"
        if os.path.exists(os.path.join(raw_dir, "train_sentences.txt")):
            build_csv_from_raw(raw_dir, data_dir)

    splits = {}
    for split in ["train", "dev", "test"]:
        path = os.path.join(data_dir, f"{split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Khong tim thay {path}. Hay chay: python data/loader_sentiment_only.py"
            )

        X, y = load_csv(path)
        splits[split] = (X, y)
        print(
            f"  [{split}] {len(X)} mau - "
            f"Neg: {y.count(0)} | Neu: {y.count(1)} | Pos: {y.count(2)}"
        )

    return splits


if __name__ == "__main__":
    raw_dir = "data/uit-vsfc-raw-sentiment"
    out_dir = "data/uit-vsfc-sentiment"

    if not os.path.exists(os.path.join(raw_dir, "train_sentences.txt")):
        download_raw(raw_dir)
    else:
        print(f"Da co raw tai {raw_dir}/, bo qua buoc tai.")

    build_csv_from_raw(raw_dir, out_dir)
    print("Xong. File nay chi xu ly sentences + sentiments, khong dung topics.")
