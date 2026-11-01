import time
import requests
from datetime import datetime, timedelta, timezone
from config import MAX_PAGES

URL = "https://gql.tokopedia.com/graphql"

PAGE_LIMIT = 50
REQUEST_DELAY = 1
REQUEST_TIMEOUT = 30

WIB = timezone(timedelta(hours=7))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "*/*",
    "Origin": "https://www.tokopedia.com",
    "Referer": "https://www.tokopedia.com/",
    "Content-Type": "application/json",
}

REVIEW_QUERY = """
query productReviewList(
  $productID: String!,
  $page: Int!,
  $limit: Int!,
  $sortBy: String,
  $filterBy: String
) {
  productrevGetProductReviewList(
    productID: $productID,
    page: $page,
    limit: $limit,
    sortBy: $sortBy,
    filterBy: $filterBy
  ) {
    hasNext
    list {
      feedbackID
      variantName
      message
      productRating
      reviewCreateTime
      reviewCreateTimestamp
      isAnonymous
      imageAttachments { attachmentID }
      videoAttachments { attachmentID }
      user { fullName }
      likeDislike { totalLike }
    }
  }
}
"""

class ScraperError(Exception):
    """Error saat mengambil ulasan; pesannya aman ditampilkan ke pengguna."""

def _build_payload(product_id, page):
    return [
        {
            "operationName": "productReviewList",
            "variables": {
                "productID": product_id,
                "page": page,
                "limit": PAGE_LIMIT,
                "sortBy": "informative_score asc",
                "filterBy": "",
            },
            "query": REVIEW_QUERY,
        }
    ]

def _to_wib_string(value):
    if value is None:
        return None

    try:
        return datetime.fromtimestamp(int(value), tz=WIB).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return None

def _parse_review(x, product_id):
    user = x.get("user") or {}
    like = x.get("likeDislike") or {}

    return {
        "review_id": x.get("feedbackID"),
        "product_id": product_id,
        "username": user.get("fullName"),
        "is_anonymous": x.get("isAnonymous"),
        "variant_name": x.get("variantName"),
        "review_timestamp": _to_wib_string(x.get("reviewCreateTime")),
        "review_age": x.get("reviewCreateTimestamp"),
        "rating": x.get("productRating"),
        "like_count": like.get("totalLike") or 0,
        "review_text": x.get("message"),
        "has_image": bool(x.get("imageAttachments")),
        "has_video": bool(x.get("videoAttachments")),
    }

def _fetch_page(session, product_id, page):
    try:
        response = session.post(
            URL,
            json=_build_payload(product_id, page),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ScraperError(f"Gagal menghubungi Tokopedia: {e}") from e

    try:
        data = response.json()
    except ValueError as e:
        raise ScraperError("Respons Tokopedia bukan JSON.") from e

    if not isinstance(data, list) or not data or data[0].get("data") is None:
        errors = (
            data[0].get("errors")
            if isinstance(data, list) and data
            else "respons kosong"
        )
        raise ScraperError(f"Tokopedia mengembalikan error: {errors}")

    result = (data[0]["data"] or {}).get("productrevGetProductReviewList")

    if result is None:
        raise ScraperError(
            "Produk tidak ditemukan atau tidak memiliki ulasan. "
            "Periksa kembali ID Produk yang dimasukkan."
        )

    return result

def fetch_reviews(product_id, max_pages=MAX_PAGES):
    product_id = str(product_id).strip()

    if not product_id.isdigit():
        raise ScraperError("Product ID harus berupa angka.")

    reviews = []
    seen_ids = set()

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for page in range(1, max_pages + 1):
            try:
                result = _fetch_page(session, product_id, page)
            except ScraperError:
                if not reviews:
                    raise
                break

            items = result.get("list") or []

            if not items:
                break

            for x in items:
                review_id = x.get("feedbackID")

                if review_id in seen_ids:
                    continue

                seen_ids.add(review_id)
                reviews.append(_parse_review(x, product_id))

            if not result.get("hasNext"):
                break

            time.sleep(REQUEST_DELAY)

    reviews.sort(
        key=lambda r: (r["review_timestamp"] is None, r["review_timestamp"] or "")
    )

    return reviews