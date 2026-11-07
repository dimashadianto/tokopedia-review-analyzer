import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer,
)

from config import (
    DATA_TRAIN_PATH,
    MODEL_RESULT_PATH,
    MIN_WORDS,
    TEST_RESULT_PATH,
    CONFUSION_MATRIX_PATH,
)

MODEL_NAME = "indobenchmark/indobert-base-p1"
RANDOM_STATE = 42
USE_CLASS_WEIGHT = True

LABEL2ID = {"palsu": 0, "asli": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

print(
    "Device:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

df = pd.read_csv(DATA_TRAIN_PATH)

df["label"] = (
    df["label"].astype(str).str.strip().str.lower()
)

df = df[df["label"].isin(LABEL2ID)]
df = df.dropna(subset=["text_model"])

df["text"] = df["text_model"].astype(str).str.strip()

df = df[
    df["text"].str.split().str.len() >= MIN_WORDS
]

df["labels"] = df["label"].map(LABEL2ID)
df = df.reset_index(drop=True)

print("Jumlah data:", len(df))
print(df["label"].value_counts())

groups = df["text"].factorize()[0]

sgkf = StratifiedGroupKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE,
)

folds = [
    idx for _, idx in sgkf.split(
        df, df["labels"], groups
    )
]

test_df = df.iloc[folds[0]].copy()
val_df = df.iloc[folds[1]].copy()

train_df = df.drop(
    index=np.concatenate([folds[0], folds[1]])
).copy()

for name, part in [
    ("Train", train_df),
    ("Val", val_df),
    ("Test", test_df),
]:
    print(
        name,
        len(part),
        part["label"].value_counts().to_dict(),
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
    )

def to_dataset(part):
    ds = Dataset.from_pandas(
        part[["text", "labels"]],
        preserve_index=False,
    )
    return ds.map(tokenize, batched=True)

train_ds, val_ds, test_ds = map(
    to_dataset,
    (train_df, val_df, test_df),
)

collator = DataCollatorWithPadding(tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

counts = np.bincount(
    train_df["labels"],
    minlength=2,
)

class_weights = torch.tensor(
    len(train_df) / (2 * counts),
    dtype=torch.float,
)

class WeightedTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=class_weights.to(
                outputs.logits.device
            ),
        )

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    p, r, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": float(f1.mean()),

        "precision_palsu": float(p[0]),
        "recall_palsu": float(r[0]),
        "f1_palsu": float(f1[0]),

        "precision_asli": float(p[1]),
        "recall_asli": float(r[1]),
        "f1_asli": float(f1[1]),
    }

args = TrainingArguments(
    output_dir=str(MODEL_RESULT_PATH),
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=RANDOM_STATE,
    report_to="none",
)

TrainerClass = (
    WeightedTrainer if USE_CLASS_WEIGHT else Trainer
)

trainer = TrainerClass(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("\nMulai training IndoBERT...")

result = trainer.train()

print(
    f"Training selesai. Loss: {result.training_loss:.4f}"
)

print(
    "\nValidation:",
    {
        k: round(v, 4)
        for k, v in trainer.evaluate(val_ds).items()
        if k.startswith("eval_")
    },
)

print(
    "Testing   :",
    {
        k: round(v, 4)
        for k, v in trainer.evaluate(test_ds).items()
        if k.startswith("eval_")
    },
)

pred = trainer.predict(test_ds)

y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=1)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Palsu", "Asli"],
        digits=4,
        zero_division=0,
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Palsu", "Asli"],
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix IndoBERT")

plt.tight_layout()

CONFUSION_MATRIX_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

plt.close()

print(
    "Confusion Matrix disimpan:",
    CONFUSION_MATRIX_PATH,
)

MODEL_RESULT_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

trainer.save_model(str(MODEL_RESULT_PATH))
tokenizer.save_pretrained(str(MODEL_RESULT_PATH))

print("\nModel disimpan:", MODEL_RESULT_PATH)

if "review_id" in test_df.columns:
    out = test_df[
        ["review_id", "text", "label"]
    ].copy()
else:
    out = test_df[
        ["text", "label"]
    ].copy()

out["prediction"] = [
    ID2LABEL[int(i)] for i in y_pred
]

out["benar"] = (
    out["label"] == out["prediction"]
)

TEST_RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    TEST_RESULT_PATH,
    index=False,
)

print("Hasil testing disimpan:", TEST_RESULT_PATH)
