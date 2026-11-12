import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MIN_WORDS = 3
MIN_PAGES = 1
MAX_PAGES = 50
DEFAULT_PAGES = 3

DETIK_PER_HALAMAN = 0.5
DETIK_PER_ULASAN = 0.1

SLANG_PATH = BASE_DIR / "training" / "data" / "indonesia_slang.csv"
DATA_TRAIN_PATH = BASE_DIR / "training" / "data" / "review_label.csv"

MODEL_RESULT_PATH = os.getenv("MODEL_ID", "dimashadianto/indobert-review")
HF_TOKEN = os.getenv("HF_TOKEN") or None

TEST_RESULT_PATH = BASE_DIR / "training" / "result" / "hasil_testing_indobert.csv"
CONFUSION_MATRIX_PATH = BASE_DIR / "training" / "result" / "confusion_matrix.png"

PER_PAGE_OPTIONS = (10, 20, 50, 100, 200, 500)