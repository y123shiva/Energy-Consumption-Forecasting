# =========================================================
# ⚡ Energy Forecasting & Monitoring Dashboard
# Production-ready Streamlit App
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
from pathlib import Path
from scipy.stats import ks_2samp


# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Forecasting & Monitoring")


# =========================================================
# Paths (Cloud Safe)
# =========================================================
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "energy_data.csv"
MODEL_PATH = ROOT / "xgb_model.pkl"


# =========================================================
# Loaders
# =========================================================
@st.cache_data
def load_baseline():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df["rolling7"] = df["Energy"].rolling(7).mean()
    return df


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_runs():
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        runs = mlflow.search_runs()

        if runs.empty:
            return pd.DataFrame()

        cols = ["run_id", "metrics.MAE", "metrics.RMSE", "metrics.MAPE"]
        runs = runs[[c for c in cols if c in runs.columns]]

        runs.rename(
            columns={
                "metrics.MAE": "MAE",
                "metrics.RMSE": "RMSE",
                "metrics.MAPE": "MAPE",
            },
            inplace=True,
        )

        return runs.sort_values("MAE")

    except Exception:
        return pd.DataFrame()


# =========================================================
# Utils
# =========================================================
def feature_drift(base, current):
    results = []

    numeric_cols = base.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if col not in current.columns:
            continue

        stat, p = ks_2samp(base[col].dropna(), current[col].dropna())

        if p < 0.01:
            level = "🔴 HIGH"
        elif p < 0.05:
            level = "🟡 MEDIUM"
        else:
            level = "🟢 LOW"

        results.append([col, stat, p, level])

    return pd.DataFrame(results, columns=["feature", "ks_stat", "p_value", "drift"])


def local_predict(file):
    model = load_model()
    if model is None:
        st.error("Model file not found.")
        return None

    df = pd.read_csv(file)
    df["prediction"] = model.predict(df)
    return df


# =========================================================
# Load Data
# =========================================================
baseline_df = load_baseline()
runs_df = load_runs()


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("⚙ Controls")

date_range = st.sidebar.date_input(
    "Date Range",
    [baseline_df["Date"].min(), baseline_df["Date"].max()],
)

show_rolling = st.sidebar.checkbox("Show 7-day average", value=True)


filtered_df = baseline_df[
    (baseline_df["Date"] >= pd.to_datetime(date_range[0]))
    & (baseline_df["Date"] <= pd.to_datetime(date_range[1]))
]


# =========================================================
# KPI Cards
# =========================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", len(filtered_df))
c2.metric("Avg Energy", round(filtered_df["Energy"].mean(), 2))
c3.metric("Max Energy", round(filtered_df["Energy"].max(), 2))
c4.metric("Min Energy", round(filtered_df["Energy"].min(), 2))


st.divider()


# =========================================================
# Tabs
# =========================================================
tabs = st.tabs(
    [
        "📈 Historical Data",
        "🚀 Predictions",
        "📉 Drift",
        "📊 Experiments",
    ]
)


# =========================================================
# TAB 1 — Historical
# =========================================================
with tabs[0]:
    st.subheader("Energy Demand Trend")

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(filtered_df["Date"], filtered_df["Energy"], label="Energy")

    if show_rolling:
        ax.plot(filtered_df["Date"], filtered_df["rolling7"], label="7-day Avg")

    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy")

    st.pyplot(fig)

    st.dataframe(filtered_df.tail(), use_container_width=True)


# =========================================================
# TAB 2 — Local Predictions
# =========================================================
with tabs[1]:
    st.subheader("Upload CSV for Prediction")

    uploaded = st.file_uploader("Inference file", type=["csv"])

    if uploaded:
        with st.spinner("Running model..."):
            preds = local_predict(uploaded)

        if preds is not None:
            st.success("Prediction completed")

            st.dataframe(preds.head(), use_container_width=True)

            fig, ax = plt.subplots()
            ax.plot(preds["prediction"])
            ax.set_title("Forecast Output")
            st.pyplot(fig)


# =========================================================
# TAB 3 — Drift
# =========================================================
with tabs[2]:
    st.subheader("Feature Drift Detection (KS Test)")

    drift_file = st.file_uploader("Upload current production CSV", type=["csv"], key="drift")

    if drift_file:
        current_df = pd.read_csv(drift_file)
        drift_df = feature_drift(baseline_df, current_df)

        st.dataframe(drift_df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.bar(drift_df["feature"], drift_df["ks_stat"])
        ax.set_xticklabels(drift_df["feature"], rotation=45)
        st.pyplot(fig)


# =========================================================
# TAB 4 — Experiments
# =========================================================
with tabs[3]:
    st.subheader("MLflow Experiment Tracking")

    if runs_df.empty:
        st.info("No MLflow runs found. Train models locally to log experiments.")
    else:
        st.dataframe(runs_df, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.bar_chart(runs_df.set_index("run_id")["MAE"])

        with c2:
            st.bar_chart(runs_df.set_index("run_id")["RMSE"])


# =========================================================
# Footer
# =========================================================
st.divider()
st.caption("Built with Streamlit • Energy Forecasting MLOps Dashboard")