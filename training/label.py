
import re
import sys
from pathlib import Path
import pandas as pd
from service.pre_process import preprocess_review_for_model

GENERIC_REVIEWS = {
    "bagus", "baguss", "good", "good product", "mantap",
    "mantul", "oke", "ok", "sesuai", "recommended",
    "rekomended", "puas", "keren", "top", "bagus sesuai",
}

PROMOTION_PATTERN = re.compile(
    r"(wa\.me|https?://|whatsapp|\bwa\b|\bdm\b|follow|"
    r"kode voucher|cek (toko|profil)|\b08\d{8,}\b)",
    re.I,
)

ASPECT_PATTERN = re.compile(
    r"(wangi|aroma|bau|tahan|ketahanan|long ?lasting|"
    r"projection|sillage|botol|packing|kemasan|pengiriman|"
    r"cepat|spray|semprot|jam|seharian|harga|original|ori|"
    r"segel|tutup|nozzle|kantor|kuliah|pagi|malam|"
    r"cowo|cewe|unisex)",
    re.I,
)

COMPARISON_PATTERN = re.compile(
    r"(tapi|namun|sayang|kurang|dibanding|lebih|"
    r"seperti|mirip|awalnya|setelah)",
    re.I,
)

def normalize_duplicate_text(text):
    text = str(text).lower().strip()
    return re.sub(r"[^a-z0-9 ]", "", text)

def has_media(value):
    if pd.isna(value):
        return False

    if isinstance(value, str):
        return value.strip().lower() in {
            "true", "1", "yes", "ya"
        }

    return bool(value)

def calculate_review_score(row, duplicate_count, burst_count):
    original_text = str(row["review_text"])
    text = original_text.lower().strip()
    words = text.split()

    fake_score = 0
    genuine_score = 0
    reasons = []

    if len(words) <= 2 or text in GENERIC_REVIEWS:
        fake_score += 1
        reasons.append("K1 terlalu pendek/generik")

    if duplicate_count >= 3 and len(words) >= 3:
        fake_score += 2
        reasons.append("K2 teks duplikat >=3")

    if (
        re.search(r"(.)\1{3,}", text)
        or (len(original_text) > 8 and original_text.isupper())
    ):
        fake_score += 1
        reasons.append("K3 huruf berulang/kapital semua")

    if PROMOTION_PATTERN.search(text):
        fake_score += 2
        reasons.append("K4 promosi/kontak/link")

    if burst_count >= 10 and len(words) <= 5:
        fake_score += 1
        reasons.append("K5 lonjakan ulasan pendek")

    if len(words) >= 10:
        genuine_score += 1
        reasons.append("A1 teks cukup panjang")

    if len(ASPECT_PATTERN.findall(text)) >= 2:
        genuine_score += 1
        reasons.append("A2 menyebut >=2 aspek")

    if COMPARISON_PATTERN.search(text):
        genuine_score += 1
        reasons.append("A3 ada pembanding/kekurangan")

    if has_media(row.get("has_image")) or has_media(row.get("has_video")):
        genuine_score += 1
        reasons.append("A4 ada foto/video")

    label = "palsu" if fake_score > genuine_score else "asli"

    return pd.Series({
        "skor_palsu": fake_score,
        "skor_asli": genuine_score,
        "label": label,
        "alasan": "; ".join(reasons),
    })

def process_reviews(source_path, output_path):
    df = pd.read_csv(source_path)

    required_columns = {
        "review_text",
        "review_id",
        "review_timestamp",
        "variant_name",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}"
        )

    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["_normalized_text"] = df["review_text"].map(
        normalize_duplicate_text
    )

    df["_duplicate_count"] = 1
    valid_reviews = df["_normalized_text"].ne("")

    df.loc[valid_reviews, "_duplicate_count"] = (
        df.loc[valid_reviews]
        .groupby("_normalized_text")["_normalized_text"]
        .transform("size")
    )

    df["_review_hour"] = pd.to_datetime(
        df["review_timestamp"],
        errors="coerce",
    ).dt.floor("h")

    df["_burst_count"] = df.groupby(
        ["_review_hour", "variant_name"],
        dropna=False,
    )["review_id"].transform("size")

    scoring_results = df.apply(
        lambda row: calculate_review_score(
            row,
            int(row["_duplicate_count"]),
            int(row["_burst_count"]),
        ),
        axis=1,
    )

    output_df = pd.concat([df, scoring_results], axis=1)

    output_df["text_model"] = output_df["review_text"].apply(
        preprocess_review_for_model
    )

    output_df.drop(
        columns=[
            "_normalized_text",
            "_duplicate_count",
            "_review_hour",
            "_burst_count",
        ],
        inplace=True,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    process_reviews(sys.argv[1], sys.argv[2])
