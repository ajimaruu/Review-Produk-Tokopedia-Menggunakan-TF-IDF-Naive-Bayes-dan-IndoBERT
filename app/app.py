import streamlit as st

st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 Aspect-Based Sentiment Analysis")

st.markdown("""
Selamat datang di aplikasi **Aspect-Based Sentiment Analysis & Named Entity Recognition**.

Project ini dibuat sebagai implementasi jurnal untuk memenuhi UAS **Natural Language Processing**.

---

### ✨ Fitur

- Aspect Based Sentiment Analysis
- Named Entity Recognition (BIO Tagging)
- TF-IDF + Naive Bayes
- IndoBERT
- Batch Prediction
- Dashboard Evaluasi

Silakan pilih menu di sidebar.
""")

st.info("Gunakan menu di sebelah kiri untuk memulai.")