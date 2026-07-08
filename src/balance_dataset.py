import pandas as pd
from pathlib import Path

# =====================================================
# PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "preprocessed.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "balanced.csv"

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 50)
print("Distribusi Awal")
print("=" * 50)
print(df["sentiment"].value_counts())

# =====================================================
# BALANCING
# =====================================================

min_count = df["sentiment"].value_counts().min()

print(f"\nJumlah data tiap kelas setelah balancing: {min_count}")

balanced = []

for label in ["positive", "negative", "neutral"]:

    sample = (
        df[df["sentiment"] == label]
        .sample(n=min_count, random_state=42)
    )

    balanced.append(sample)

balanced_df = pd.concat(
    balanced,
    ignore_index=True
)

# Acak dataset
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\n")
print("=" * 50)
print("Distribusi Baru")
print("=" * 50)
print(balanced_df["sentiment"].value_counts())

# =====================================================
# SAVE
# =====================================================

balanced_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nDataset berhasil disimpan:")
print(OUTPUT_FILE)