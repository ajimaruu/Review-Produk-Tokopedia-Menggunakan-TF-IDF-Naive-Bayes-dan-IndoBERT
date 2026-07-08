from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About Project")

st.markdown("""
# 🛒 Tokopedia Review Insight

### Aspect-Based Sentiment Analysis (ABSA) & Named Entity Recognition (NER)

Aplikasi ini merupakan implementasi **Natural Language Processing (NLP)** untuk menganalisis review pelanggan e-commerce menggunakan kombinasi:

- 🇮🇩 IndoBERT
- 🤖 Naive Bayes
- 🏷 Named Entity Recognition (NER)
- 📌 Aspect-Based Sentiment Analysis (ABSA)

Project ini dibuat sebagai implementasi penelitian mengenai analisis sentimen berbasis aspek pada review pelanggan Tokopedia.
""")

st.divider()

# =============================================================

st.header("🎯 Tujuan")

st.markdown("""
Project ini bertujuan untuk:

- Mengidentifikasi sentiment review pelanggan.
- Menemukan aspek yang dibahas pelanggan.
- Membandingkan performa Naive Bayes dan IndoBERT.
- Menampilkan visualisasi hasil analisis.
- Membantu memahami opini pelanggan terhadap suatu produk.
""")

st.divider()

# =============================================================

st.header("⚙️ Workflow")

st.code("""
Review
   │
   ▼
Preprocessing
   │
   ▼
Case Folding
   │
   ▼
Normalization
   │
   ▼
Stopword Removal
   │
   ▼
Stemming
   │
   ▼
Aspect Detection
   │
   ▼
NER
   │
   ▼
Sentiment Classification
   │
   ▼
Visualization
""")

st.divider()

# =============================================================

st.header("🧠 Machine Learning Models")

c1, c2 = st.columns(2)

with c1:

    st.subheader("🤖 Naive Bayes")

    st.markdown("""
Model klasik Machine Learning yang menggunakan:

- TF-IDF Vectorizer
- Multinomial Naive Bayes

Kelebihan:

- Sangat cepat
- Ringan
- Cocok untuk dataset besar

Kekurangan:

- Tidak memahami konteks kalimat.
""")

with c2:

    st.subheader("🇮🇩 IndoBERT")

    st.markdown("""
Model Transformer pretrained Bahasa Indonesia.

Kelebihan:

- Memahami konteks kalimat.
- Akurasi tinggi.
- Cocok untuk NLP modern.

Kekurangan:

- Lebih lambat dibanding Naive Bayes.
- Membutuhkan resource lebih besar.
""")

st.divider()

# =============================================================

st.header("🏷 Aspect yang Didukung")

aspect = [
    "Harga",
    "Pengiriman",
    "Kemasan",
    "Pelayanan",
    "Kualitas",
    "Keaslian",
    "Produk",
    "Penjual"
]

cols = st.columns(4)

for i, asp in enumerate(aspect):
    cols[i % 4].success(asp)

st.divider()

# =============================================================

st.header("🛠 Technology Stack")

tech = {
    "Programming Language": "Python",
    "Framework": "Streamlit",
    "Machine Learning": "Scikit-Learn",
    "Deep Learning": "Transformers (HuggingFace)",
    "Language Model": "IndoBERT",
    "NER": "BERT Token Classification",
    "Visualization": "Plotly",
    "Dataset": "Tokopedia Review Dataset"
}

st.table(tech)

st.divider()

# =============================================================

st.header("📂 Project Structure")

st.code("""
app/
│
├── app.py
├── pages/
│
src/
│
├── preprocessing.py
├── prepare_dataset.py
├── train_absa.py
├── train_bert.py
├── train_ner.py
├── predict.py
│
models/
│
├── absa_nb_model.pkl
├── tfidf_vectorizer.pkl
├── bert/
│
data/
│
├── raw/
├── processed/
│
reports/
""")

st.divider()

# =============================================================

st.header("📈 Dataset")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Language",
    "Indonesia"
)

c2.metric(
    "Domain",
    "E-Commerce"
)

c3.metric(
    "Source",
    "Tokopedia"
)

st.divider()

# =============================================================

st.header("👨‍🎓 Author")

st.info("""
**Nama :** Aji Seno

**Mata Kuliah :** Natural Language Processing

**Universitas :** Universitas Dian Nuswantoro

**Tahun :** 2026
""")

st.divider()

st.success(
    "Terima kasih telah menggunakan aplikasi Tokopedia Review Insight."
)