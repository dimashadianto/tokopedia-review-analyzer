import threading
from pathlib import Path
import numpy as np
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import MODEL_RESULT_PATH, HF_TOKEN
import re

MAX_LENGTH = 128
BATCH_SIZE = 16

ID_PALSU = 0
ID_ASLI = 1
LABELS = {ID_PALSU: "palsu", ID_ASLI: "asli"}

RGB_PALSU = (220, 38, 38)
RGB_ASLI = (22, 163, 74)

_infer_lock = threading.RLock()

tokenizer = AutoTokenizer.from_pretrained(MODEL_RESULT_PATH, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_RESULT_PATH, token=HF_TOKEN
)
model.eval()

def _predict_proba(texts):
    texts = [str(t) for t in texts]
    all_probs = []

    with _infer_lock:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]

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

        results.append(
            {
                "label": LABELS[label_id],
                "prob_asli": round(float(row[ID_ASLI]), 4),
                "prob_palsu": round(float(row[ID_PALSU]), 4),
            }
        )

    return results

_COLOR_RE = re.compile(
    r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)"
    r"|#([0-9a-fA-F]{6})\b"
)

def _retint(r, g, b):
    if max(r, g, b) - min(r, g, b) < 15:
        return None

    if r >= b:
        base, strength = RGB_PALSU, (255 - g) / 255
    else:
        base, strength = RGB_ASLI, (255 - r) / 255

    strength = min(max(strength, 0.0), 1.0)
    return tuple(int(round(255 - strength * (255 - c))) for c in base)

def _recolor_match(m):
    if m.group(5):
        h = m.group(5)
        new = _retint(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        return m.group(0) if new is None else "#%02x%02x%02x" % new

    r, g, b = (float(m.group(i)) for i in (1, 2, 3))
    new = _retint(r, g, b)

    if new is None:
        return m.group(0)

    alpha = m.group(4)
    if alpha is not None:
        return "rgba(%d,%d,%d,%s)" % (new[0], new[1], new[2], alpha)
    return "rgb(%d,%d,%d)" % new

def _recolor_html(html):
    return _COLOR_RE.sub(_recolor_match, html)

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
    shap_values = _get_explainer()([text])
    sv = shap_values[0][:, ID_PALSU]

    html = _recolor_html(shap.plots.text(sv, display=False))

    values = np.asarray(sv.values, dtype=float)
    base_value = float(np.asarray(sv.base_values).ravel()[0])

    tokens = [
        {"token": token.strip(), "value": float(value)}
        for token, value in zip(sv.data, values)
        if token.strip()
    ]

    return {
        "html": html,
        "target_label": LABELS[ID_PALSU],
        "base_value": base_value,
        "output_value": base_value + float(np.sum(values)),
        "tokens": tokens,
    }
