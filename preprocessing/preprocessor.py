import re
import unicodedata


TEENCODE_MAP = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hok": "không",
    "hong": "không",
    "kg": "không",
    "khum": "không",
    "dc": "được",
    "dk": "được",
    "đc": "được",
    "bt": "bình thường",
    "bth": "bình thường",
    "mn": "mọi người",
    "sv": "sinh viên",
    "gv": "giảng viên",
    "qt": "quá trời",
}


VIETNAMESE_STOPWORDS = {
    "và", "của", "là", "có", "được", "cho", "với", "các", "trong",
    "này", "một", "đã", "những", "từ", "tôi", "thì", "mà", "vì",
    "nếu", "khi", "để", "về", "cũng", "đó", "lại", "bị", "đến",
    "nên", "đây", "còn", "hơn", "như", "ra", "vào", "lên", "xuống",
    "theo", "rồi", "thôi", "thế", "vậy", "ấy", "đâu", "sao", "mình",
    "em", "anh", "chị", "bạn", "họ", "chúng",
}

# Không xóa các từ này vì chúng rất quan trọng cho sentiment.
SENTIMENT_KEEP_WORDS = {
    "không", "chưa", "chẳng", "đừng", "rất", "quá", "hơi", "kém",
    "tốt", "hay", "tệ", "dở", "chán", "khó", "dễ", "ổn",
}

_word_tokenize = None
_underthesea_ok = None


def _get_word_tokenize():
    """Lazy import underthesea để nếu chưa cài thì chương trình vẫn chạy được."""
    global _word_tokenize, _underthesea_ok
    if _underthesea_ok is None:
        try:
            from underthesea import word_tokenize
            _word_tokenize = word_tokenize
            _underthesea_ok = True
        except ImportError:
            _underthesea_ok = False
            print("Chưa cài underthesea -> bỏ qua bước ghép từ.")
    return _word_tokenize if _underthesea_ok else None


class Preprocessor:
    def __init__(
        self,
        normalize_unicode=True,
        normalize_teencode=True,
        remove_punct=True,
        remove_digits=True,
        word_segment=True,
        remove_stopwords=False,
    ):
        self.normalize_unicode = normalize_unicode
        self.normalize_teencode = normalize_teencode
        self.remove_punct = remove_punct
        self.remove_digits = remove_digits
        self.word_segment = word_segment
        self.remove_stopwords = remove_stopwords

    def clean(self, text):
        """Xử lý 1 câu và trả về chuỗi đã làm sạch."""
        if not isinstance(text, str):
            return ""

        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)

        if self.normalize_teencode:
            text = self._normalize_teencode(text)

        if self.remove_punct:
            text = re.sub(r"[^\w\s]", " ", text)

        if self.remove_digits:
            text = re.sub(r"\d+", " ", text)

        text = re.sub(r"\s+", " ", text).strip()

        if self.word_segment and text:
            word_tokenize = _get_word_tokenize()
            if word_tokenize is not None:
                text = word_tokenize(text, format="text")

        if self.remove_stopwords and text:
            text = self._remove_stopwords(text)

        return text

    def clean_batch(self, texts):
        """Xử lý danh sách câu."""
        return [self.clean(text) for text in texts]

    def _normalize_teencode(self, text):
        words = text.split()
        normalized = [TEENCODE_MAP.get(word, word) for word in words]
        return " ".join(normalized)

    def _remove_stopwords(self, text):
        words = []
        for word in text.split():
            plain_word = word.replace("_", " ")
            if word in SENTIMENT_KEEP_WORDS or plain_word in SENTIMENT_KEEP_WORDS:
                words.append(word)
            elif word not in VIETNAMESE_STOPWORDS and plain_word not in VIETNAMESE_STOPWORDS:
                words.append(word)
        return " ".join(words)


if __name__ == "__main__":
    prep = Preprocessor()
    examples = [
        "gv rất dễ hiểu!!!",
        "Môn này ko dễ hiểu.",
        "Sv thấy bài tập quá khó.",
        "xem tại https://example.com <b>rất hữu ích</b>",
    ]

    for example in examples:
        print("Input :", example)
        print("Output:", prep.clean(example))
        print()
