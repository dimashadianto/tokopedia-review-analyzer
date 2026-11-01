import math
import os
import threading
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from config import MIN_WORDS, MAX_PAGES
from service.model import explain_review, predict_batch
from service.preprocess import split_reviews
from service.scraper import MAX_PAGES, PAGE_LIMIT, ScraperError, fetch_reviews
from store import TTL_SECONDS, get_result, save_result

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "...")

_shap_lock = threading.Lock()

def hitung_skor(kept):
    if not kept:
        return 0.0

    jumlah_asli = sum(1 for r in kept if r["label"] == "asli")
    return round(100 * jumlah_asli / len(kept), 1)

def kategori_rekomendasi(skor):
    if skor >= 80:
        return "Layak Dipertimbangkan"
    if skor >= 50:
        return "Perlu Dipertimbangkan Kembali"
    return "Tidak Direkomendasikan"

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/analisis")
def analisis():
    pid = request.form.get("product_id", "").strip()

    if not pid.isdigit():
        flash("Product ID harus berupa angka.")
        return redirect(url_for("index"))

    try:
        reviews = fetch_reviews(pid)
    except ScraperError as e:
        flash(str(e))
        return redirect(url_for("index"))

    if not reviews:
        flash("Produk ini belum punya ulasan.")
        return redirect(url_for("index"))

    kept, removed = split_reviews(reviews)

    if not kept:
        flash(f"Semua ulasan kurang dari {MIN_WORDS} kata, tidak ada yang bisa dianalisis.")
        return redirect(url_for("index"))

    predictions = predict_batch([r["review_clean"] for r in kept])

    for review, pred in zip(kept, predictions):
        review.update(pred)

    skor = hitung_skor(kept)
    jumlah_palsu = sum(1 for r in kept if r["label"] == "palsu")

    save_result(
        pid,
        kept=kept,
        removed=removed,
        total_scraped=len(reviews),
        jumlah_asli=len(kept) - jumlah_palsu,
        jumlah_palsu=jumlah_palsu,
        skor=skor,
        kategori=kategori_rekomendasi(skor),
    )

    return redirect(url_for("hasil", pid=pid))


@app.get("/hasil/<pid>")
def hasil(pid):
    data = get_result(pid)

    if data is None:
        flash("Data tidak ditemukan atau sudah kedaluwarsa, silakan analisis ulang.")
        return redirect(url_for("index"))

    total = len(data["kept"])
    total_pages = max(1, math.ceil(total / MAX_PAGES))
    page = min(max(request.args.get("page", 1, type=int), 1), total_pages)

    start = (page - 1) * MAX_PAGES
    end = min(start + MAX_PAGES, total)

    items = [(i, data["kept"][i]) for i in range(start, end)]

    return render_template(
        "hasil.html",
        pid=pid,
        data=data,
        items=items,
        page=page,
        total_pages=total_pages,
        min_words=MIN_WORDS,
    )


@app.get("/detail/<pid>/<int:idx>")
def detail(pid, idx):
    data = get_result(pid)

    if data is None:
        flash("Data tidak ditemukan atau sudah kedaluwarsa, silakan analisis ulang.")
        return redirect(url_for("index"))

    if not 0 <= idx < len(data["kept"]):
        abort(404)

    review = data["kept"][idx]

    if "shap" not in review:
        with _shap_lock:
            if "shap" not in review:
                review["shap"] = explain_review(review["review_clean"])

    tokens = review["shap"]["tokens"]
    top_tokens = sorted(tokens, key=lambda t: abs(t["value"]), reverse=True)[:8]
    max_abs = max((abs(t["value"]) for t in tokens), default=0) or 1.0

    return render_template(
        "detail.html",
        pid=pid,
        idx=idx,
        review=review,
        top_tokens=top_tokens,
        max_abs=max_abs,
    )


@app.get("/panduan")
def panduan():
    return render_template(
        "panduan.html",
        min_words=MIN_WORDS,
        ttl_minutes=TTL_SECONDS // 60,
        max_reviews=MAX_PAGES * PAGE_LIMIT,
    )


@app.get("/tentang")
def tentang():
    return render_template("tentang.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)