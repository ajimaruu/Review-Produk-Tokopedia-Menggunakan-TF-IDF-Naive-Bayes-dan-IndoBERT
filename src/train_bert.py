# src/train_bert.py

import pandas as pd

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# =====================
# 1. Load Dataset
# =====================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "balanced.csv"
)

print(df.head())


# =====================
# 2. Cleaning Dataset
# =====================

# pastikan teks tidak kosong

df["after"] = (
    df["after"]
    .fillna("")
    .astype(str)
)


# normalisasi sentiment

df["sentiment"] = (
    df["sentiment"]
    .str.lower()
    .str.strip()
)


print(
    df["sentiment"].unique()
)



# =====================
# 3. Encode Label
# =====================

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


df["label"] = df["sentiment"].map(
    label_map
)


# cek label gagal

print(
    "Label kosong:",
    df["label"].isna().sum()
)


print(
    df[df["label"].isna()].head()
)


# hapus data tanpa label

df = df.dropna(
    subset=["label"]
)


# ubah ke integer

df["label"] = (
    df["label"]
    .astype(int)
)


print(
    df["label"].value_counts()
)



# =====================
# 4. Dataset HuggingFace
# =====================

dataset = Dataset.from_pandas(
    df
)


dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

print(dataset)



# =====================
# 5. Load Tokenizer
# =====================

model_name = (
    "indobenchmark/indobert-base-p1"
)


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)



# =====================
# 6. Tokenisasi
# =====================

def tokenize(batch):

    return tokenizer(

        batch["after"],

        truncation=True,

        padding="max_length",

        max_length=128

    )



dataset = dataset.map(
    tokenize,
    batched=True
)



# hapus kolom tidak diperlukan

remove_columns = [
    "text",
    "aspect",
    "sentiment",
    "before",
    "after"
]


for col in remove_columns:

    if col in dataset["train"].column_names:

        dataset = dataset.remove_columns(
            col
        )



# format torch

dataset.set_format(
    "torch"
)


print(
    dataset["train"][0]
)



# =====================
# 7. Load Model IndoBERT
# =====================

model = AutoModelForSequenceClassification.from_pretrained(

    model_name,

    num_labels=3,

    id2label={
        0:"Negative",
        1:"Neutral",
        2:"Positive"
    },

    label2id={
        "Negative":0,
        "Neutral":1,
        "Positive":2
    }

)


model.config.problem_type = (
    "single_label_classification"
)



# =====================
# 8. Training Arguments
# =====================

training_args = TrainingArguments(

    output_dir="models/bert",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=4,

    num_train_epochs=5,

    weight_decay=0.01,

    logging_steps=100,
    
    save_strategy="epoch"

)



# =====================
# 9. Trainer
# =====================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    eval_dataset=dataset["test"]

)



# cek label

print(
    "Contoh label:",
    dataset["train"][0]["label"]
)


print(
    type(dataset["train"][0]["label"])
)



# =====================
# 10. Training
# =====================

trainer.train()



# =====================
# 11. Save Model
# =====================

trainer.save_model(
    "models/bert"
)


tokenizer.save_pretrained(
    "models/bert"
)



# =====================
# 12. Evaluasi & Metrik
# =====================
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Lakukan prediksi pada data testing
print("\nMenghitung prediksi...")
result = trainer.predict(dataset["test"])

# Ambil logits (prediksi mentah) dan label asli
logits = result.predictions
y_true = result.label_ids

# Ubah probabilitas menjadi kelas prediksi final (0, 1, atau 2)
y_pred = np.argmax(logits, axis=-1)

# Hitung semua metrik
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\n" + "="*35)
print("🏆 HASIL EVALUASI MODEL INDOBERT")
print("="*35)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")

print("\n📊 CLASSIFICATION REPORT:")
# Nama target disesuaikan dengan id2label di kodemu
print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))

print("\n🧮 CONFUSION MATRIX:")
print(confusion_matrix(y_true, y_pred))