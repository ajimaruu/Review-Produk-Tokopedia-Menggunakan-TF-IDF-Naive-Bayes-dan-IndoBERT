from pathlib import Path
import sys
import io
import time
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.predict import predict

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Batch Prediction")

st.markdown("""
Upload file **CSV** yang berisi review pelanggan.

Kolom yang wajib ada:

- **text**
""")

st.divider()

# =====================================================
# MODEL
# =====================================================

model = st.radio(
    "Model",
    ["IndoBERT", "Naive Bayes"],
    horizontal=True
)

# =====================================================
# UPLOAD
# =====================================================

uploaded = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded is not None:

    df = pd.read_csv(uploaded)

    st.success(f"Berhasil membaca {len(df)} review.")

    st.subheader("Preview Dataset")

    st.dataframe(df.head(), use_container_width=True)

    if "text" not in df.columns:

        st.error("CSV harus memiliki kolom bernama 'text'")

        st.stop()

    st.divider()

    if st.button("🚀 Mulai Prediksi", use_container_width=True):

        progress = st.progress(0)

        status = st.empty()

        results = []

        start = time.time()

        total = len(df)

        for i, row in df.iterrows():

            res = predict(
                str(row["text"]),
                model
            )

            results.append({

                "text": row["text"],

                "preprocessed": res["preprocessed_text"],

                "sentiment": res["sentiment"],

                "confidence": round(res["confidence"],2),

                "aspect": ", ".join(res["aspect"]),

                "prediction_time": res["prediction_time"]

            })

            progress.progress((i+1)/total)

            status.text(f"Processing {i+1}/{total}")

        end = time.time()

        result_df = pd.DataFrame(results)

        # =======================================
        # SAVE FOR DASHBOARD
        # =======================================

        report_dir = ROOT / "reports"
        report_dir.mkdir(exist_ok=True)

        report_file = report_dir / "prediction_result.csv"

        result_df.to_csv(
            report_file,
            index=False,
            encoding="utf-8-sig"
        )

        st.success("Hasil berhasil disimpan untuk Dashboard.")

        st.success("Prediksi selesai.")

        st.divider()
        # =======================================
        # METRIC
        # =======================================

        positive = (result_df.sentiment=="Positive").sum()

        negative = (result_df.sentiment=="Negative").sum()

        neutral = (result_df.sentiment=="Neutral").sum()

        avg_conf = result_df.confidence.mean()

        c1,c2,c3,c4=st.columns(4)

        c1.metric("Positive",positive)

        c2.metric("Negative",negative)

        c3.metric("Neutral",neutral)

        c4.metric("Avg Confidence",f"{avg_conf:.2f}%")

        st.divider()

        st.subheader("Prediction Result")

        st.dataframe(
            result_df,
            use_container_width=True
        )

        # =======================================
        # DOWNLOAD
        # =======================================

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(

            "⬇ Download Result",

            csv,

            "prediction_result.csv",

            "text/csv",

            use_container_width=True

        )

        st.info(
            f"""
Total Prediction : {len(result_df)}

Execution Time : {end-start:.2f} second
"""
        )

