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
from config import (
    MIN_WORDS, MAX_PAGES, MIN_PAGES, DEFAULT_PAGES,
    PER_PAGE_OPTIONS, DETIK_PER_HALAMAN, DETIK_PER_ULASAN,
)
from service.model import explain_review, predict_batch
from service.preprocess import split_reviews
from service.scraper import PAGE_LIMIT, ScraperError, extract_product_id, fetch_reviews
from store import (
    TTL_SECONDS,
    get_kept,
    get_kept_page,
    get_summary,
    init_db,
    save_result,
    save_shap,
)

import sqlite3
c = sqlite3.connect("data/hasil.db")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "...")

init_db()

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
    return render_template(
        "index.html",
        min_pages=MIN_PAGES,
        max_pages=MAX_PAGES,
        default_pages=DEFAULT_PAGES,
        page_limit=PAGE_LIMIT,
        detik_per_halaman=DETIK_PER_HALAMAN,
        detik_per_ulasan=DETIK_PER_ULASAN,
        min_words=MIN_WORDS,
    )

@app.post("/analisis")
def analisis():
    link = request.form.get("product_link", "").strip()
    jumlah_halaman = request.form.get("jumlah_halaman", DEFAULT_PAGES, type=int)
    jumlah_halaman = max(MIN_PAGES, min(jumlah_halaman, MAX_PAGES))

    try:
        pid = extract_product_id(link)
        reviews = fetch_reviews(pid, max_pages=jumlah_halaman)
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
    data = get_summary(pid)

    if data is None:
        flash("Data tidak ditemukan atau sudah kedaluwarsa, silakan analisis ulang.")
        return redirect(url_for("index"))

    total = data["total_kept"]

    per_page = request.args.get("per_page", 10, type=int)
    if per_page not in PER_PAGE_OPTIONS:
        per_page = 10

    total_pages = max(1, math.ceil(total / per_page))
    page = min(max(request.args.get("page", 1, type=int), 1), total_pages)

    start = (page - 1) * per_page

    items = get_kept_page(pid, start, per_page)

    return render_template(
        "hasil.html",
        pid=pid,
        data=data,
        items=items,
        n=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        min_words=MIN_WORDS,
    )

@app.get("/detail/<pid>/<int:idx>")
def detail(pid, idx):
    data = get_summary(pid)

    if data is None:
        flash("Data tidak ditemukan atau sudah kedaluwarsa, silakan analisis ulang.")
        return redirect(url_for("index"))

    review = get_kept(pid, idx)

    if review is None:
        abort(404)

    if "shap" not in review:
        with _shap_lock:
            review = get_kept(pid, idx)
            if review is None:
                abort(404)
            if "shap" not in review:
                review["shap"] = explain_review(review["review_clean"])
                save_shap(pid, idx, review["shap"])

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