import json
import os
import sqlite3
import time
from contextlib import contextmanager

TTL_SECONDS = 30 * 60

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hasil.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hasil (
    pid           TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    total_scraped INTEGER NOT NULL,
    jumlah_asli   INTEGER NOT NULL,
    jumlah_palsu  INTEGER NOT NULL,
    skor          REAL NOT NULL,
    kategori      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ulasan (
    pid   TEXT NOT NULL REFERENCES hasil(pid) ON DELETE CASCADE,
    kind  TEXT NOT NULL CHECK (kind IN ('kept', 'removed')),
    idx   INTEGER NOT NULL,
    data  TEXT NOT NULL,   -- JSON isi ulasan + prediksi
    shap  TEXT,            -- JSON hasil SHAP, diisi saat halaman detail dibuka
    PRIMARY KEY (pid, kind, idx)
);

CREATE INDEX IF NOT EXISTS idx_hasil_created ON hasil(created_at);
"""

@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _db() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)

def purge_expired():
    batas = time.time() - TTL_SECONDS
    with _db() as conn:
        conn.execute("DELETE FROM hasil WHERE created_at < ?", (batas,))

def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, default=float)

def save_result(pid, *, kept, removed, total_scraped,
                jumlah_asli, jumlah_palsu, skor, kategori):
    purge_expired()
    with _db() as conn:
        conn.execute("DELETE FROM hasil WHERE pid = ?", (pid,))
        conn.execute(
            "INSERT INTO hasil (pid, created_at, total_scraped, jumlah_asli, "
            "jumlah_palsu, skor, kategori) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, time.time(), total_scraped, jumlah_asli,
             jumlah_palsu, skor, kategori),
        )
        conn.executemany(
            "INSERT INTO ulasan (pid, kind, idx, data) VALUES (?, 'kept', ?, ?)",
            ((pid, i, _dumps(r)) for i, r in enumerate(kept)),
        )
        conn.executemany(
            "INSERT INTO ulasan (pid, kind, idx, data) VALUES (?, 'removed', ?, ?)",
            ((pid, i, _dumps(r)) for i, r in enumerate(removed)),
        )

def get_summary(pid):
    with _db() as conn:
        row = conn.execute("SELECT * FROM hasil WHERE pid = ?", (pid,)).fetchone()
        if row is None:
            return None
        if time.time() - row["created_at"] > TTL_SECONDS:
            conn.execute("DELETE FROM hasil WHERE pid = ?", (pid,))
            return None
        data = dict(row)
        jumlah = conn.execute(
            "SELECT kind, COUNT(*) AS n FROM ulasan WHERE pid = ? GROUP BY kind",
            (pid,),
        ).fetchall()
        hitung = {r["kind"]: r["n"] for r in jumlah}
        data["total_kept"] = hitung.get("kept", 0)
        data["total_removed"] = hitung.get("removed", 0)
        return data

def get_kept_page(pid, offset, limit):
    with _db() as conn:
        rows = conn.execute(
            "SELECT idx, data FROM ulasan "
            "WHERE pid = ? AND kind = 'kept' AND idx >= ? AND idx < ? "
            "ORDER BY idx",
            (pid, offset, offset + limit),
        ).fetchall()
    return [(r["idx"], json.loads(r["data"])) for r in rows]

def get_kept(pid, idx):
    with _db() as conn:
        row = conn.execute(
            "SELECT data, shap FROM ulasan "
            "WHERE pid = ? AND kind = 'kept' AND idx = ?",
            (pid, idx),
        ).fetchone()
    if row is None:
        return None
    review = json.loads(row["data"])
    if row["shap"]:
        review["shap"] = json.loads(row["shap"])
    return review

def save_shap(pid, idx, shap):
    with _db() as conn:
        conn.execute(
            "UPDATE ulasan SET shap = ? WHERE pid = ? AND kind = 'kept' AND idx = ?",
            (_dumps(shap), pid, idx),
        )

def get_removed(pid):
    with _db() as conn:
        rows = conn.execute(
            "SELECT data FROM ulasan WHERE pid = ? AND kind = 'removed' ORDER BY idx",
            (pid,),
        ).fetchall()
    return [json.loads(r["data"]) for r in rows]