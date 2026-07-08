from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Hasil Prediksi")

# =====================================================
# LOAD DATA
# =====================================================

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "prediction_result.csv"

if not REPORT.exists():
    st.warning("""
Belum ada hasil Batch Prediction.

Silakan buka menu **Batch Prediction** terlebih dahulu.
""")
    st.stop()

try:
    df = pd.read_csv(REPORT)

    if df.empty:
        st.warning("File hasil prediksi masih kosong.")
        st.stop()

except Exception as e:
    st.error(f"Gagal membaca file:\n{e}")
    st.stop()

# =====================================================
# METRICS
# =====================================================

total = len(df)

positive = (df["sentiment"] == "Positive").sum()
negative = (df["sentiment"] == "Negative").sum()
neutral = (df["sentiment"] == "Neutral").sum()

avg_conf = df["confidence"].mean()

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Review", total)
c2.metric("😊 Positive", positive)
c3.metric("😐 Neutral", neutral)
c4.metric("😡 Negative", negative)
c5.metric("Confidence", f"{avg_conf:.2f}%")

st.divider()

# =====================================================
# CHART
# =====================================================

left, right = st.columns(2)

with left:

    st.subheader("Distribusi Sentimen")

    sentiment_count = (
        df["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_count.columns = ["Sentiment", "Jumlah"]

    fig = px.pie(
        sentiment_count,
        values="Jumlah",
        names="Sentiment",
        hole=.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    st.subheader("Jumlah Sentimen")

    fig2 = px.bar(
        sentiment_count,
        x="Sentiment",
        y="Jumlah",
        text="Jumlah"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.divider()

# =====================================================
# ASPECT
# =====================================================

if "aspect" in df.columns:

    st.subheader("Distribusi Aspect")

    aspects = []

    for row in df["aspect"].fillna(""):

        for asp in str(row).split(","):

            asp = asp.strip()

            if asp != "":
                aspects.append(asp)

    if len(aspects) > 0:

        aspect_df = (
            pd.Series(aspects)
            .value_counts()
            .reset_index()
        )

        aspect_df.columns = [
            "Aspect",
            "Jumlah"
        ]

        fig3 = px.bar(
            aspect_df,
            x="Aspect",
            y="Jumlah",
            text="Jumlah"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

st.divider()

# =====================================================
# CONFIDENCE
# =====================================================

st.subheader("Confidence Distribution")

fig4 = px.histogram(
    df,
    x="confidence",
    nbins=20
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

st.divider()

# =====================================================
# FILTER
# =====================================================

st.subheader("Filter Data")

option = st.selectbox(
    "Sentiment",
    ["All"] + sorted(df["sentiment"].unique().tolist())
)

if option != "All":
    filtered = df[df["sentiment"] == option]
else:
    filtered = df

# =====================================================
# TABLE
# =====================================================

st.subheader("Prediction Result")

st.dataframe(
    filtered,
    use_container_width=True,
    height=450
)

# =====================================================
# DOWNLOAD
# =====================================================

csv = filtered.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "⬇ Download CSV",
    csv,
    "dashboard_result.csv",
    "text/csv",
    use_container_width=True
)