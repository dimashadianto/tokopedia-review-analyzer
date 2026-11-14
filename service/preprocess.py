import csv
import itertools
import re

from config import (
    MIN_WORDS,
    SLANG_PATH,
    ABBREVIATION_PATH,
    ELONGATION_PATH,
)

PREFIX_PATTERN = re.compile(
    r"^(kualitas produk|product quality)\s*:\s*",
    re.IGNORECASE,
)

WORD_PATTERN = re.compile(r"[a-z0-9]+")
ELONGATED_PATTERN = re.compile(r"([a-z])\1\1")

def load_dictionary(path):
    dictionary = {}

    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            slang = (row.get("slang") or "").strip().lower()
            formal = (row.get("formal") or "").strip().lower()

            if slang and formal and WORD_PATTERN.fullmatch(slang):
                dictionary.setdefault(slang, formal)

    return dictionary

ELONGATION_DICTIONARY = load_dictionary(ELONGATION_PATH)
ABBREVIATION_DICTIONARY = load_dictionary(ABBREVIATION_PATH)
SLANG_DICTIONARY = load_dictionary(SLANG_PATH)

KNOWN_WORDS = (
    set(ELONGATION_DICTIONARY.values())
    | set(ABBREVIATION_DICTIONARY.values())
    | set(SLANG_DICTIONARY.values())
    | set(ABBREVIATION_DICTIONARY)
    | set(SLANG_DICTIONARY)
)

def _is_empty(text):
    return text is None or (isinstance(text, float) and text != text)

def remove_review_prefix(text):
    if _is_empty(text):
        return ""

    return PREFIX_PATTERN.sub("", str(text).strip())

def clean_review_text(text):
    text = remove_review_prefix(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()

def apply_case_folding(text):
    return text.lower().strip()

def _squeeze(word, max_repeat):
    return "".join(
        "".join(group)[:max_repeat]
        for _, group in itertools.groupby(word)
    )

def squeeze_elongation(word):
    if not ELONGATED_PATTERN.search(word):
        return word

    for limit in (1, 2):
        candidate = _squeeze(word, limit)
        if candidate in KNOWN_WORDS:
            return candidate

    return _squeeze(word, 1)


def normalize_word(word):
    # 1. pemanjangan karakter
    if word in ELONGATION_DICTIONARY:
        return ELONGATION_DICTIONARY[word]

    word = squeeze_elongation(word)

    # 2. singkatan
    if word in ABBREVIATION_DICTIONARY:
        return ABBREVIATION_DICTIONARY[word]

    # 3. slang
    return SLANG_DICTIONARY.get(word, word)


def normalize_text(text):
    return " ".join(normalize_word(word) for word in text.split())


def count_words(text):
    if not text:
        return 0

    return len(text.split())

def preprocess_review(text):
    text = clean_review_text(text)
    text = apply_case_folding(text)
    text = normalize_text(text)

    return text

def preprocess_review_for_model(text):
    processed_text = preprocess_review(text)

    if count_words(processed_text) < MIN_WORDS:
        return None

    return processed_text

def split_reviews(reviews, min_words=MIN_WORDS):
    kept = []
    removed = []

    for review in reviews:
        item = dict(review)

        raw_text = item.get("review_text")
        item["review_text_raw"] = "" if _is_empty(raw_text) else str(raw_text)
        item["review_text"] = remove_review_prefix(raw_text)
        item["review_clean"] = preprocess_review(item["review_text"])
        item["word_count"] = count_words(item["review_clean"])

        if item["word_count"] < min_words:
            item["removed_reason"] = f"kurang dari {min_words} kata"
            removed.append(item)
        else:
            kept.append(item)

    return kept, removed
