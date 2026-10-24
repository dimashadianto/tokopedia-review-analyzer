
import re
import csv
from config import SLANG_PATH, MIN_WORDS

PREFIX_PATTERN = re.compile(
    r"^(kualitas produk|product quality)\s*:\s*",
    re.IGNORECASE,
)

def load_slang_dictionary(path):
    slang_dictionary = {}

    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            slang = (row.get("slang") or "").strip().lower()
            formal = (row.get("formal") or "").strip().lower()

            if slang and formal:
                slang_dictionary.setdefault(slang, formal)

    return slang_dictionary


SLANG_DICTIONARY = load_slang_dictionary(SLANG_PATH)

def remove_review_prefix(text):
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

def normalize_slang(text):
    return " ".join(
        SLANG_DICTIONARY.get(word, word)
        for word in text.split()
    )

def count_words(text):
    if not text:
        return 0

    return len(text.split())

def preprocess_review(text):
    text = clean_review_text(text)
    text = apply_case_folding(text)
    text = normalize_slang(text)

    return text

def preprocess_review_for_model(text):
    processed_text = preprocess_review(text)

    if count_words(processed_text) < MIN_WORDS:
        return None

    return processed_text
