import threading
from pathlib import Path
import numpy as np
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import MODEL_RESULT_PATH

MAX_LENGTH = 128
BATCH_SIZE = 16

ID_PALSU = 0
ID_ASLI = 1
LABELS = {ID_PALSU: "palsu", ID_ASLI: "asli"}

_infer_lock = threading.RLock()

if not Path(MODEL_RESULT_PATH).exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan di {MODEL_RESULT_PATH}. "
        "Jalankan training/train.py dulu atau cek MODEL_RESULT_PATH di config.py."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_RESULT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_RESULT_PATH)
model.eval()

def _predict_proba(texts):
    texts = [str(t) for t in texts]
    all_probs = []

    with _infer_lock:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            with torch.inference_mode():
                logits = model(**encoded).logits

            all_probs.append(torch.softmax(logits, dim=-1).numpy())

    if not all_probs:
        return np.empty((0, 2))

    return np.concatenate(all_probs, axis=0)

def predict_batch(texts):
    probs = _predict_proba(texts)
    results = []

    for row in probs:
        label_id = int(np.argmax(row))

        results.append({
            "label": LABELS[label_id],
            "prob_asli": round(float(row[ID_ASLI]), 4),
            "prob_palsu": round(float(row[ID_PALSU]), 4),
        })

    return results

_explainer = None

def _get_explainer():
    global _explainer

    if _explainer is None:
        masker = shap.maskers.Text(tokenizer)
        _explainer = shap.Explainer(
            _predict_proba,
            masker,
            output_names=[LABELS[ID_PALSU], LABELS[ID_ASLI]],
        )

    return _explainer


def explain_review(text):
    probs = _predict_proba([text])[0]
    target = int(np.argmax(probs))

    shap_values = _get_explainer()([text])

    raw_tokens = shap_values.data[0]
    values = shap_values.values[0][:, target]
    base_value = float(np.asarray(shap_values.base_values[0])[target])

    tokens = [
        {"token": token.strip(), "value": float(value)}
        for token, value in zip(raw_tokens, values)
        if token.strip()
    ]

    return {
        "target_label": LABELS[target],
        "base_value": base_value,
        "output_value": base_value + float(np.sum(values)),
        "tokens": tokens,
    }