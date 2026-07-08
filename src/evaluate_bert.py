import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


print("START EVALUATION")


# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(
    "data/processed/preprocessed.csv"
)

print("Dataset loaded")
print(df.head())


# ==========================
# Label Encoding
# ==========================

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


df["sentiment"] = (
    df["sentiment"]
    .str.lower()
    .str.strip()
)


df["label"] = df["sentiment"].map(label_map)


# buang data yang label kosong

df = df.dropna(
    subset=["label"]
)


df["label"] = df["label"].astype(int)


df["after"] = (
    df["after"]
    .fillna("")
    .astype(str)
)


print("Jumlah data:", len(df))


# ==========================
# Dataset HuggingFace
# ==========================

dataset = Dataset.from_pandas(
    df
)


dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)


print(dataset)


# ==========================
# Load Model
# ==========================

model_path = "models/bert"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_path
)


print("Loading model...")


model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)


print("Model loaded")


# ==========================
# Tokenisasi
# ==========================

def tokenize(batch):

    return tokenizer(
        batch["after"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


print("Tokenizing...")


dataset = dataset.map(
    tokenize,
    batched=True
)


# hapus kolom tidak diperlukan

dataset = dataset.remove_columns(
    [
        "text",
        "aspect",
        "sentiment",
        "before",
        "after",
    ]
)


dataset.set_format(
    "torch"
)


print(dataset["test"][0])


# ==========================
# Evaluasi
# ==========================

print("Predicting...")


trainer = Trainer(
    model=model
)


result = trainer.predict(
    dataset["test"]
)


print("Prediction selesai")


# ==========================
# Metric
# ==========================

y_true = result.label_ids


y_pred = result.predictions.argmax(
    axis=1
)


accuracy = accuracy_score(
    y_true,
    y_pred
)


precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted"
)


print("\n======================")
print("HASIL EVALUASI BERT")
print("======================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


print("\nClassification Report\n")


print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Negative",
            "Neutral",
            "Positive"
        ]
    )
)


print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


print("\nSELESAI")