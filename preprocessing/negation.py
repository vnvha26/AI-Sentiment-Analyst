NEGATION_WORDS = {
    "không",
    "chưa",
    "chẳng",
    "chả",
    "khong",
    "chua",
}

NEGATION_TARGETS = {
    "tốt",
    "hay",
    "dễ",
    "hiểu",
    "vui",
    "thích",
    "ổn",
    "nhiệt_tình",
    "tận_tình",
    "tận_tâm",
    "vui_vẻ",
    "vui_tính",
    "hiệu_quả",
    "chi_tiết",
    "thoải_mái",
}

NEGATION_TARGET_PHRASES = {
    ("dễ", "hiểu"),
    ("hài", "lòng"),
    ("nhiệt", "tình"),
    ("tận", "tình"),
    ("tận", "tâm"),
    ("vui", "vẻ"),
    ("vui", "tính"),
    ("hiệu", "quả"),
    ("chi", "tiết"),
    ("thoải", "mái"),
}


def normalize_negation(text):
    words = text.split()
    normalized = []
    index = 0

    while index < len(words):
        word = words[index]

        if word in NEGATION_WORDS and index + 1 < len(words):
            next_word = words[index + 1]

            if index + 2 < len(words):
                phrase = (next_word, words[index + 2])
                if phrase in NEGATION_TARGET_PHRASES:
                    normalized.append(f"{word}_{phrase[0]}_{phrase[1]}")
                    index += 3
                    continue

            if next_word in NEGATION_TARGETS:
                normalized.append(f"{word}_{next_word}")
                index += 2
                continue

        normalized.append(word)
        index += 1

    return " ".join(normalized)
