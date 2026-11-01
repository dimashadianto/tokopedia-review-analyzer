from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MIN_WORDS = 3
MAX_PAGES = 20

SLANG_PATH = BASE_DIR / "training" / "data" / "indonesia_slang.csv"
DATA_TRAIN_PATH = BASE_DIR / "training" / "data" / "review_label.csv"
MODEL_RESULT_PATH = BASE_DIR / "model" / "indobert-review"
TEST_RESULT_PATH = BASE_DIR / "training" / "results" / "hasil_testing_indobert.csv"
