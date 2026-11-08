import re
import sys
from pathlib import Path
import pandas as pd
from service.preprocess import preprocess_review_for_model

DUPLICATE_MIN_COUNT = 10
BURST_MIN_COUNT = 10
SHORT_REVIEW_MAX_WORDS = 5
GENERIC_RATIO_MIN = 0.6
TEMPLATE_PREFIX_WORDS = 6
TEMPLATE_MIN_COUNT = 5
REPETITIVE_DIVERSITY_MAX = 0.5
LONG_TEXT_MIN_WORDS = 10
NATURAL_MIN_WORDS = 6
NATURAL_DIVERSITY_MIN = 0.6

ACCOUNT_COLUMN_CANDIDATES = (
    "reviewer_id", "user_id", "account_id", "reviewer_name", "username",
)

def _compile(pattern):
    return re.compile(pattern, re.I)

def _words(*items):
    return r"\b(?:" + "|".join(items) + r")\b"

GENERIC_WORDS = {
    "bagus", "baguss", "good", "mantap", "mantab", "mantul", "oke", "ok",
    "sip", "josss", "jos", "keren", "top", "best", "nice", "great",
    "sesuai", "recommended", "rekomended", "rekomen", "puas",
    "barang", "produk", "banget", "sekali", "sangat", "pokoknya",
    "deskripsi", "gambar", "seller", "penjual", "terima", "kasih",
    "makasih", "thanks", "thx", "product",
    "jelek", "buruk", "zonk", "kecewa", "parah", "penipu", "tipu",
    "sampah", "kapok", "payah",
}

PROMOTION_PATTERN = _compile(
    r"(wa\.me|t\.me|https?://|www\.|\bwhatsapp\b|\bwa\b|\bdm\b|"
    r"\bfollow\b|kode voucher|cek (?:toko|profil|etalase|bio)|"
    r"kunjungi|reseller|dropship|grosir|@\w{3,}|"
    r"\b(?:mending|lebih bagus|lebih murah)\s+(?:beli\s+)?"
    r"(?:di|ke)\s+(?:toko|sebelah|lapak))"
)

PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?62|0)8\d{8,11}(?!\d)")

PURCHASE_CALL_PATTERN = _compile(
    r"\b(?:ayo|yuk|buruan|segera|silakan|jangan ragu(?: untuk)?|wajib|"
    r"harus|mending)\s+(?:langsung\s+)?(?:beli|order|checkout|co|borong|"
    r"pesan|coba|punya)\b"
    r"|\b(?:beli|order|checkout)\s+(?:sekarang|aja|saja)\b"
    r"|\bjangan (?:pernah )?(?:beli|order|tertipu|percaya)\b|\bhindari\b"
    r"|" + _words("serbu", "borong", "mumpung", "dijamin")
)

EXAGGERATION_PATTERN = _compile(
    r"\bterbaik\b|\bnomor (?:1|satu)\b|\bno\.? ?1\b|"
    r"\bpaling (?:bagus|wangi|mantap|top|keren)\b|\b100 ?%|"
    r"\bseratus persen\b|\bluar biasa\b|\bsempurna\b|\bdahsyat\b|"
    r"\bajaib\b|\b(?:tiada|tidak ada|gak ada) (?:tanding|lawan)\b|"
    r"\bsepanjang masa\b|"
    r"\bterburuk\b|\bsampah\b|\bpenipu\b|\bscam\b|\bkapok\b"
)

REASON_PATTERN = _compile(
    _words("karena", "karna", "krn", "soalnya", "sebab", "makanya",
           "sehingga", "alasannya", "gara-gara", "gara gara",
           "supaya", "biar", "buktinya")
)

EXPERIENCE_PATTERN = _compile(
    r"\b(?:saya|aku|gue|gw|ku)\s+(?:sudah|udah|pakai|pake|beli|order|"
    r"coba|nyoba|suka|pesan|dapat|dapet|pernah|sempat|baru)\b"
    r"|\b(?:sudah|udah|baru)\s+(?:pakai|pake|beli|coba|nyoba|dipakai|"
    r"dicoba|order|repeat)\b"
    r"|\b(?:dipakai|dipake|dicoba|kupakai|kubeli|pemakaian|memakai|"
    r"menggunakan|mencoba|nyoba|awalnya|ternyata)\b"
    r"|\brepeat order\b|\b(?:beli|order) lagi\b"
    r"|\b(?:kali|kalinya) (?:kedua|ketiga|keempat)\b"
    r"|\bsetelah (?:dipakai|pakai|dicoba|disemprot|semprot|seharian)\b"
    r"|\b(?:pas|waktu|saat) (?:dipakai|dicoba|disemprot|buka|nyoba)\b"
)

CONTEXT_PATTERN = _compile(
    _words("kantor", "ngantor", "kuliah", "kampus", "sekolah", "kerja",
           "olahraga", "gym", "hangout", "nongkrong", "kondangan",
           "acara", "pesta", "kencan", "perjalanan", "hujan", "panas",
           "dingin", "ruangan", "ac", "cuaca", "siang", "sore", "pagi",
           "malam", "seharian", "harian", "santai", "formal", "casual",
           "outdoor", "indoor", "musim", "sehari-hari", "sehari hari")
)

PREFERENCE_PATTERN = _compile(
    r"\bmenurut(?:ku| saya| aku)?\b|\bkurasa\b|\bkupikir\b|\bjujur\b|"
    r"\b(?:saya|aku|gue|gw)\s+(?:kurang suka|lebih suka|suka|pikir|rasa)\b|"
    r"\b(?:buat|untuk|bagi)\s+(?:aku|saya|ku)\b|"
    r"\b(?:kurang )?cocok (?:buat|untuk|di|dengan|sama)\b|"
    r"\bselera\b|\bfavorit\b"
)

COMPARISON_PATTERN = _compile(
    _words("tapi", "namun", "sayang", "sayangnya", "kurang", "cuma",
           "kekurangan", "kelemahan", "minus", "dibanding",
           "dibandingkan", "dibandingin", "ketimbang", "mirip",
           "beda", "berbeda")
    + r"|\blebih (?:\w+ )?(?:dari|daripada|dibanding)\b"
)

ASPECT_GROUPS = {
    "aroma": _compile(_words("wangi", "wanginya", "aroma", "harum", "bau",
                             "manis", "segar", "fresh", "woody", "floral",
                             "citrus", "vanilla", "musk", "powdery",
                             "menyengat", "lembut")),
    "ketahanan": _compile(r"\b(?:tahan|ketahanan|awet|long ?lasting)\b"
                          r"|\b\d+\s?jam\b"),
    "proyeksi": _compile(_words("projection", "proyeksi", "sillage",
                                "jejak")),
    "kemasan": _compile(_words("botol", "packing", "kemasan", "bungkus",
                               "segel", "tutup", "nozzle", "spray",
                               "semprot", "semprotan", "dus", "box")),
    "pengiriman": _compile(_words("pengiriman", "kirim", "dikirim",
                                  "ekspedisi", "kurir")),
    "harga": _compile(_words("harga", "murah", "mahal", "worth")),
    "keaslian": _compile(_words("original", "ori", "asli", "palsu", "kw")),
}

def reduce_elongation(text):
    return re.sub(r"([a-z])\1{2,}", r"\1", text)

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text)

def normalize_duplicate_text(text):
    text = reduce_elongation(str(text).lower())
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def has_media(value):
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "ya"}
    return bool(value)

def has_promotion(text):
    compact = re.sub(r"(?<=\d)[ .\-](?=\d)", "", text)
    return bool(PROMOTION_PATTERN.search(text) or PHONE_PATTERN.search(compact))

def calculate_review_score(row, duplicate_count, burst_count, template_count):
    original_text = str(row["review_text"])
    text = original_text.lower().strip()
    clean = reduce_elongation(text)
    words = tokenize(clean)
    n = len(words)

    generic_ratio = (
        sum(w in GENERIC_WORDS for w in words) / n if n else 1.0
    )
    diversity = len(set(words)) / n if n else 0.0
    aspects = [k for k, p in ASPECT_GROUPS.items() if p.search(clean)]

    has_experience = bool(EXPERIENCE_PATTERN.search(clean))
    has_reason = bool(REASON_PATTERN.search(clean))
    has_context = bool(CONTEXT_PATTERN.search(clean))
    has_preference = bool(PREFERENCE_PATTERN.search(clean))
    has_comparison = bool(COMPARISON_PATTERN.search(clean))
    has_photo_video = (
        has_media(row.get("has_image")) or has_media(row.get("has_video"))
    )

    fake, genuine, codes, reasons = 0, 0, [], []

    def add_fake(points, code, reason):
        nonlocal fake
        fake += points
        codes.append(code)
        reasons.append(f"{code} {reason}")

    def add_genuine(code, reason):
        nonlocal genuine
        genuine += 1
        codes.append(code)
        reasons.append(f"{code} {reason}")

    if n <= 2 or (n >= 3 and generic_ratio >= GENERIC_RATIO_MIN and not aspects):
        add_fake(1, "K1", "terlalu pendek/pujian generik tanpa substansi")

    if duplicate_count >= DUPLICATE_MIN_COUNT and n >= 3:
        add_fake(2, "K2", f"teks duplikat >={DUPLICATE_MIN_COUNT}")

    if re.search(r"([a-z])\1{2,}", text) or (
        len(original_text) > 8 and original_text.isupper()
    ):
        add_fake(1, "K3", "huruf berulang/kapital semua")

    if has_promotion(text):
        add_fake(2, "K4", "promosi/kontak/link")

    if burst_count >= BURST_MIN_COUNT and n <= SHORT_REVIEW_MAX_WORDS:
        add_fake(1, "K5", "lonjakan ulasan pendek")

    if (
        EXAGGERATION_PATTERN.search(clean)
        and not has_reason
        and len(aspects) < 2
    ):
        add_fake(1, "K6", "klaim berlebihan tanpa penjelasan")

    if PURCHASE_CALL_PATTERN.search(clean):
        add_fake(1, "K7", "ajakan membeli/slogan")

    is_repetitive = n >= 6 and diversity < REPETITIVE_DIVERSITY_MAX
    is_template = (
        template_count >= TEMPLATE_MIN_COUNT
        and duplicate_count < DUPLICATE_MIN_COUNT
    )
    if is_repetitive or is_template:
        add_fake(1, "K8", "pola repetitif/template")

    substance = (
        has_experience or has_reason or has_context or has_preference
        or has_comparison or has_photo_video or bool(aspects)
    )

    if fake > 0 and n >= 3 and not substance:
        add_fake(1, "K9", "tidak ada informasi pengalaman bermakna")

    if n >= LONG_TEXT_MIN_WORDS and generic_ratio < 0.5:
        add_genuine("A1", "teks cukup panjang")

    if len(aspects) >= 2:
        add_genuine("A2", f"menyebut >=2 aspek ({', '.join(aspects)})")

    if has_comparison:
        add_genuine("A3", "ada pembanding/kekurangan")

    if has_photo_video:
        add_genuine("A4", "ada foto/video")

    if has_experience:
        add_genuine("A5", "menceritakan pengalaman penggunaan")

    if has_reason:
        add_genuine("A6", "ada alasan di balik penilaian")

    if has_context:
        add_genuine("A7", "ada konteks/kondisi penggunaan")

    if has_preference:
        add_genuine("A8", "menunjukkan preferensi/perspektif pribadi")

    if (
        n >= NATURAL_MIN_WORDS
        and diversity >= NATURAL_DIVERSITY_MIN
        and generic_ratio < 0.5
        and substance
        and not any(c in codes for c in ("K3", "K4", "K8"))
    ):
        add_genuine("A9", "bahasa natural dan relevan")

    label = "palsu" if fake > genuine else "asli"

    return {
        "skor_palsu": fake,
        "skor_asli": genuine,
        "label": label,
        "kode_indikator": ";".join(codes),
        "alasan": "; ".join(reasons),
    }

def find_account_column(df):
    for col in ACCOUNT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None

def add_dataset_features(df):
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["_normalized_text"] = df["review_text"].map(normalize_duplicate_text)
    non_empty = df["_normalized_text"].ne("")

    df["_duplicate_count"] = 1
    account_col = find_account_column(df)
    grouped = df.loc[non_empty].groupby("_normalized_text")
    if account_col:
        df.loc[non_empty, "_duplicate_count"] = grouped[account_col].transform(
            "nunique"
        )
    else:
        df.loc[non_empty, "_duplicate_count"] = grouped[
            "_normalized_text"
        ].transform("size")

    timestamp = pd.to_datetime(df["review_timestamp"], errors="coerce")
    df["_review_hour"] = timestamp.dt.floor("h")
    df["_is_short"] = (
        df["_normalized_text"].str.split().str.len().fillna(0)
        <= SHORT_REVIEW_MAX_WORDS
    )
    variant = df["variant_name"].fillna("(tanpa varian)")
    df["_burst_count"] = (
        df.groupby([df["_review_hour"], variant])["_is_short"].transform("sum")
    )
    df.loc[df["_review_hour"].isna(), "_burst_count"] = 0

    prefix = df["_normalized_text"].map(
        lambda t: " ".join(t.split()[:TEMPLATE_PREFIX_WORDS])
        if len(t.split()) >= TEMPLATE_PREFIX_WORDS else ""
    )
    df["_template_count"] = 0
    has_prefix = prefix.ne("")
    df.loc[has_prefix, "_template_count"] = (
        prefix[has_prefix].groupby(prefix[has_prefix]).transform("size")
    )
    return df

def process_reviews(source_path, output_path):
    df = pd.read_csv(source_path)

    required_columns = {
        "review_text", "review_id", "review_timestamp", "variant_name",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    df = add_dataset_features(df)

    results = [
        calculate_review_score(
            row,
            int(row["_duplicate_count"]),
            int(row["_burst_count"]),
            int(row["_template_count"]),
        )
        for _, row in df.iterrows()
    ]
    output_df = pd.concat([df, pd.DataFrame(results, index=df.index)], axis=1)

    output_df["text_model"] = output_df["review_text"].apply(
        preprocess_review_for_model
    )
    output_df = output_df.drop(
        columns=[
            "_normalized_text", "_duplicate_count", "_review_hour",
            "_is_short", "_burst_count", "_template_count",
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(output_df["label"].value_counts().to_string())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    process_reviews(sys.argv[1], sys.argv[2])
