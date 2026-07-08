from pathlib import Path
import sys
import time
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.predict import predict

# ============================================
# CSS
# ============================================

st.markdown("""
<style>

.result-card{
    background:#1e1e1e;
    padding:18px;
    border-radius:15px;
    border:1px solid #333;
    margin-bottom:15px;
}

.metric-title{
    color:#999;
    font-size:14px;
}

.metric-value{
    font-size:34px;
    font-weight:bold;
}

.aspect-badge{
    display:inline-block;
    background:#198754;
    color:white;
    padding:8px 15px;
    border-radius:30px;
    margin:5px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# TITLE
# ============================================

st.title("🔍 Single Review Prediction")

st.caption(
    "Analisis Aspect-Based Sentiment Analysis menggunakan Naive Bayes dan IndoBERT."
)

st.divider()

# ============================================
# MODEL
# ============================================

model = st.radio(
    "Pilih Model",
    ["IndoBERT", "Naive Bayes"],
    horizontal=True
)

# ============================================
# INPUT
# ============================================

review = st.text_area(
    "Masukkan Review",
    height=170,
    placeholder="Contoh : Packing bagus tetapi pengiriman lama..."
)

# ============================================
# BUTTON
# ============================================

if st.button("🚀 Analyze", use_container_width=True):

    if review.strip() == "":
        st.warning("Masukkan review terlebih dahulu.")
        st.stop()

    with st.spinner("Menganalisis review..."):

        result = predict(review, model)

        time.sleep(0.5)

    st.success("Analisis selesai.")

    st.divider()

    # ============================================
    # SENTIMENT CARD
    # ============================================

    if result["sentiment"] == "Positive":
        sentiment_icon = "😊"
        color = "green"

    elif result["sentiment"] == "Negative":
        sentiment_icon = "😠"
        color = "red"

    else:
        sentiment_icon = "😐"
        color = "orange"

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(f"""
        <div class="result-card">

        <div class="metric-title">
        Sentiment
        </div>

        <div class="metric-value">
        {sentiment_icon} {result['sentiment']}
        </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="result-card">

        <div class="metric-title">
        Confidence
        </div>

        <div class="metric-value">
        {result['confidence']:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.progress(result["confidence"] / 100)

    with c3:

        st.markdown(f"""
        <div class="result-card">

        <div class="metric-title">
        Prediction Time
        </div>

        <div class="metric-value">
        {result['prediction_time']:.3f}s
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ============================================
    # PREPROCESS + ASPECT
    # ============================================

    left, right = st.columns(2)

    with left:

        st.subheader("🧹 Preprocessing")

        st.code(result["preprocessed_text"])

        st.subheader("🏷 Aspect")

        if len(result["aspect"]) == 0:

            st.warning("Tidak ditemukan aspect.")

        else:

            html = ""

            for asp in result["aspect"]:

                html += f"""
                <span class="aspect-badge">
                {asp}
                </span>
                """

            st.markdown(html, unsafe_allow_html=True)

    # ============================================
    # NER
    # ============================================

    with right:

        st.subheader("📌 Named Entity Recognition")

        df = pd.DataFrame(result["entities"])

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # ============================================
    # JSON
    # ============================================

    with st.expander("🔎 Detail Prediction (JSON)"):

        st.json(result)