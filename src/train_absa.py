import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split


# ======================
# Load Dataset
# ======================

print("Membaca dataset...")

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "balanced.csv"
)

print(f"Jumlah data awal : {len(df)}")

print("\nJumlah NaN:")
print(df.isna().sum())

# Hapus data kosong
df = df.dropna(subset=["after", "sentiment"])

# Pastikan string
df["after"] = df["after"].astype(str)

# Hapus string kosong
df = df[df["after"].str.strip() != ""]

print(f"\nJumlah data setelah dibersihkan : {len(df)}")

# ======================
# Feature & Label
# ======================

X = df["after"]
y = df["sentiment"]

# ======================
# TF-IDF
# ======================

print("\nMelakukan TF-IDF...")

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

print("TF-IDF selesai.")

# ======================
# Split Data
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Data Train : {X_train.shape[0]}")
print(f"Data Test  : {X_test.shape[0]}")

# ======================
# Model
# ======================

print("\nTraining Naive Bayes...")

model = ComplementNB()
model.fit(X_train, y_train)

print("Training selesai.")

# ======================
# Prediction
# ======================

pred = model.predict(X_test)

# =====================
# Evaluasi & Metrik Naive Bayes
# =====================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Pastikan variabel model kamu bernama 'model' atau sesuaikan dengan namamu (misal: 'nb_model', 'clf', dll)
# Pastikan juga fitur test bernama 'X_test' (atau 'X_test_tfidf') dan label test bernama 'y_test'

print("\nMenghitung prediksi Naive Bayes...")
y_pred = model.predict(X_test) # Sesuaikan nama variabel jika berbeda

# Hitung semua metrik menggunakan mode 'macro' agar rata-rata dihitung merata untuk semua kelas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\n" + "="*35)
print("🏆 HASIL EVALUASI MODEL NAIVE BAYES")
print("="*35)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")

print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\n🧮 CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

# ======================
# Confusion Matrix
# ======================

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

# ======================
# Save Model
# ======================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/absa_nb_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel berhasil disimpan.")

# ======================
# Error Analysis
# ======================

os.makedirs("reports", exist_ok=True)

# Ambil kembali text sesuai index y_test
hasil = pd.DataFrame({
    "Text": df.loc[y_test.index, "text"].values,
    "Actual": y_test.values,
    "Prediction": pred
})

# Ambil yang salah prediksi
error = hasil[hasil["Actual"] != hasil["Prediction"]].copy()

# Tambahkan kolom penyebab
error["Penyebab"] = ""

# Simpan
error.to_csv("reports/error_analysis.csv", index=False, encoding="utf-8-sig")

print(f"\nJumlah salah prediksi : {len(error)}")
print("Error analysis disimpan di reports/error_analysis.csv")

print("\nContoh error:")
print(error.head())

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Pembagian Data Latih dan Data Uji (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_label, test_size=0.2, random_state=42
)
print(f"Data Train : {X_train.shape[0]}")
print(f"Data Test  : {X_test.shape[0]}")

# 2. Implementasi dan Pelatihan Model Multinomial Naive Bayes
print("Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

print("Training selesai.")