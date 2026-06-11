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


SENTIMENT_KEEP_WORDS = {
    "không", "chưa", "chẳng", "đừng", "rất", "quá", "hơi", "kém",
    "tốt", "hay", "tệ", "dở", "chán", "khó", "dễ", "ổn",
}


_word_tokenize = None
_underthesea_ok = None


def _get_word_tokenize():
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
        normalize_special_tokens=True,
        normalize_teencode=True,
        remove_punct=True,
        remove_digits=True,
        word_segment=True,
        remove_stopwords=False,
    ):
        self.normalize_unicode = normalize_unicode
        self.normalize_special_tokens = normalize_special_tokens
        self.normalize_teencode = normalize_teencode
        self.remove_punct = remove_punct
        self.remove_digits = remove_digits
        self.word_segment = word_segment
        self.remove_stopwords = remove_stopwords

    def clean(self, text):
        if not isinstance(text, str):
            return ""

        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        text = text.lower()

        # Xóa HTML và URL
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)

        # Chuẩn hóa token đặc biệt của UIT-VSFC
        if self.normalize_special_tokens:
            text = self._normalize_special_tokens(text)

        # Chuẩn hóa teencode
        if self.normalize_teencode:
            text = self._normalize_teencode(text)

        # Xóa dấu câu
        if self.remove_punct:
            text = re.sub(r"[^\w\s]", " ", text)

        # Xóa số
        if self.remove_digits:
            text = re.sub(r"\d+", " ", text)

        # Gọn khoảng trắng
        text = re.sub(r"\s+", " ", text).strip()

        # Ghép từ tiếng Việt
        if self.word_segment and text:
            word_tokenize = _get_word_tokenize()
            if word_tokenize is not None:
                text = word_tokenize(text, format="text")

        # Xóa stopwords nếu bật
        if self.remove_stopwords and text:
            text = self._remove_stopwords(text)

        return text

    def clean_batch(self, texts):
        return [self.clean(text) for text in texts]

    def _normalize_teencode(self, text):
        for teen_word, normal_word in TEENCODE_MAP.items():
            text = re.sub(
                rf"\b{re.escape(teen_word)}\b",
                f" {normal_word} ",
                text
            )
        return text

    def _normalize_special_tokens(self, text):
        # Mã ẩn danh tên người/trường: wzjwz208, wzjwz324...
        text = re.sub(r"\bwzjwz\d+\b", " name_token ", text)

        # Số thập phân dạng 1dot5, 2dot0...
        text = re.sub(r"\b\d+dot\d+\b", " num_token ", text)

        # Emoji tích cực
        text = re.sub(
            r"\bcolon(love|lovelove|smile|smilesmile|bigsmile|smallsmile|hihi|cc)\b",
            " emoji_positive ",
            text
        )

        # Emoji tiêu cực
        text = re.sub(
            r"\bcolon(sad|sadcolon|contemn)\b",
            " emoji_negative ",
            text
        )

        # Các emoji khác
        text = re.sub(r"\bcolon[a-z]+\b", " emoji_token ", text)

        # Token thay cho dấu câu
        text = re.sub(r"\bdotdotdot\b", " ", text)
        text = re.sub(r"\bdoubledot\b", " ", text)
        text = re.sub(r"\bdot\b", " ", text)

        # v.v
        text = re.sub(r"\bvdotv\b", " etc_token ", text)

        return text

    def _remove_stopwords(self, text):
        words = []
        for word in text.split():
            plain_word = word.replace("_", " ")
            if word in SENTIMENT_KEEP_WORDS or plain_word in SENTIMENT_KEEP_WORDS:
                words.append(word)
            elif word not in VIETNAMESE_STOPWORDS and plain_word not in VIETNAMESE_STOPWORDS:
                words.append(word)
        return " ".join(words)