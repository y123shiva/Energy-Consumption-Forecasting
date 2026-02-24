import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.stats import ks_2samp
import mlflow
from io import StringIO

# ---------------------------------------------------
# Config
# ---------------------------------------------------
API_URL = "http://localhost:8000/predict/upload_async"
BASELINE_PATH = "data/energy_data.csv"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
st.set_page_config(layout="wide")

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

@st.cache_data
def load_baseline():
    df = pd.read_csv(BASELINE_PATH, parse_dates=["Date"])
    df["rolling7"] = df["Energy"].rolling(7).mean()
    return df


@st.cache_data
def load_runs():
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


# ------------------------
# Drift Detection (KS-test)
# ------------------------
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


# ------------------------
# FastAPI call
# ------------------------
def call_api(file):
    files = {"file": file}
    r = requests.post(API_URL, files=files)

    if r.status_code != 200:
        st.error(f"API error: {r.text}")
        return None

    return pd.read_csv(StringIO(r.text))


# ---------------------------------------------------
# Load baseline + experiments
# ---------------------------------------------------
baseline_df = load_baseline()
runs_df = load_runs()

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("⚡ Energy Forecasting & Monitoring Dashboard")

tabs = st.tabs(
    [
        "📈 Data",
        "🚀 Live Predictions",
        "📉 Drift Monitoring",
        "📊 Experiments",
        "🏆 Best Model",
    ]
)

# ===================================================
# TAB 1 — Historical Data
# ===================================================
with tabs[0]:
    st.subheader("Energy Demand")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(baseline_df["Date"], baseline_df["Energy"], label="Energy")
    ax.plot(baseline_df["Date"], baseline_df["rolling7"], label="7d Avg")
    ax.legend()

    st.pyplot(fig)
    st.dataframe(baseline_df.tail(), width="stretch")


# ===================================================
# TAB 2 — Live Predictions (FastAPI)
# ===================================================
with tabs[1]:
    st.subheader("Upload CSV for Live Prediction")

    uploaded = st.file_uploader("Upload inference CSV", type=["csv"])

    if uploaded:
        with st.spinner("Calling FastAPI..."):
            preds = call_api(uploaded)

        if preds is not None:
            st.success("Predictions received")

            st.dataframe(preds.head(), width="stretch")

            if "prediction" in preds.columns:
                fig, ax = plt.subplots()
                ax.plot(preds["prediction"])
                ax.set_title("Forecast Output")
                st.pyplot(fig)


# ===================================================
# TAB 3 — Drift Monitoring
# ===================================================
with tabs[2]:
    st.subheader("Feature-level Drift Detection (KS-test)")

    drift_file = st.file_uploader("Upload current production CSV", type=["csv"], key="drift")

    if drift_file:
        current_df = pd.read_csv(drift_file)

        drift_df = feature_drift(baseline_df, current_df)

        st.dataframe(drift_df, width="stretch")

        st.download_button(
            "Download drift scores",
            drift_df.to_csv(index=False),
            "drift_scores.csv",
        )

        st.subheader("Drift Visualization")

        fig, ax = plt.subplots()
        ax.bar(drift_df["feature"], drift_df["ks_stat"])
        ax.set_xticklabels(drift_df["feature"], rotation=45, ha="right")
        st.pyplot(fig)


# ===================================================
# TAB 4 — MLflow Experiments
# ===================================================
with tabs[3]:
    st.subheader("Model Comparison (MLflow)")

    if runs_df.empty:
        st.info("Run training first: python train.py")
    else:
        st.dataframe(runs_df, width="stretch")

        c1, c2 = st.columns(2)

        with c1:
            st.bar_chart(runs_df.set_index("run_id")["MAE"])

        with c2:
            st.bar_chart(runs_df.set_index("run_id")["RMSE"])


# ===================================================
# TAB 5 — Best Model
# ===================================================
with tabs[4]:
    st.subheader("Best Model (Lowest MAE)")

    if not runs_df.empty:
        best = runs_df.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric("Run ID", best["run_id"])
        c2.metric("MAE", round(best["MAE"], 2))
        c3.metric("RMSE", round(best["RMSE"], 2))

        st.success("Automatically selected best model")
