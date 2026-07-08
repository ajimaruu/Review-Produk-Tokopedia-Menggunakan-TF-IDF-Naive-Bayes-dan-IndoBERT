import pandas as pd

# ==========================
# Load Dataset
# ==========================

review_df = pd.read_csv("data/absa_dataset.csv")

aspect_df = pd.read_csv("data/aspect_dictionary.csv")

# ==========================
# Buat dictionary aspect
# ==========================

aspect_dict = {}

for _, row in aspect_df.iterrows():
    aspect_dict[row["keyword"].lower()] = row["aspect"]

# ==========================
# Cari aspect dari review
# ==========================

rows = []

for _, row in review_df.iterrows():

    text = str(row["review_text"])

    sentiment = row["sentiment_label"]

    lower = text.lower()

    found = False

    for keyword, aspect in aspect_dict.items():

        if keyword in lower:

            rows.append({

                "text": text,

                "aspect": aspect,

                "sentiment": sentiment

            })

            found = True

    if not found:

        rows.append({

            "text": text,

            "aspect": "General",

            "sentiment": sentiment

        })

# ==========================
# Simpan
# ==========================

absa = pd.DataFrame(rows)

absa.to_csv(

    "data/processed/absa_ready.csv",

    index=False,

    encoding="utf-8"

)

print(absa.head())
print()
print("Jumlah data:", len(absa))

# ==========================
# Generate NER Dataset
# ==========================

ner_lines = []

for _, row in absa.iterrows():

    text = row["text"]

    aspect = row["aspect"]

    words = text.split()

    for word in words:

        if word.lower() == aspect.lower():

            ner_lines.append(f"{word}\tB-ASPECT")

        else:

            ner_lines.append(f"{word}\tO")

    ner_lines.append("")

with open(

    "data/processed/ner_ready.tsv",

    "w",

    encoding="utf-8"

) as f:

    f.write("\n".join(ner_lines))

print("NER Dataset berhasil dibuat")