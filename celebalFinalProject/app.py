
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT_DIR / "Lead Scoring.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
DEFAULT_PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.joblib"

TIER_ORDER = {"Hot": 0, "Warm": 1, "Cold": 2}


@st.cache_resource
def load_artifacts(model_path: str, preprocessor_path: str):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def bucket_score(probability: float) -> str:
    if probability >= 0.7:
        return "Hot"
    if probability >= 0.4:
        return "Warm"
    return "Cold"


def suggested_action(tier: str) -> str:
    if tier == "Hot":
        return "Immediate Call"
    if tier == "Warm":
        return "Nurture Email"
    return "No Action"


def resolve_lead_id(df: pd.DataFrame) -> pd.Series:
    if "Prospect ID" in df.columns:
        return df["Prospect ID"]
    if "Lead Number" in df.columns:
        return df["Lead Number"]
    return pd.Series(np.arange(1, len(df) + 1), index=df.index, name="Lead ID")


def score_dataframe(df: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    scored_input = df.copy()
    if "Converted" in scored_input.columns:
        scored_input = scored_input.drop(columns=["Converted"])

    probabilities = model.predict_proba(preprocessor.transform(scored_input))[:, 1]

    out = df.copy()
    out.insert(0, "Lead ID", resolve_lead_id(df))
    out["Predicted Probability"] = probabilities
    out["Lead Score"] = (out["Predicted Probability"] * 100).round().astype(int)
    out["Lead Tier"] = out["Predicted Probability"].apply(bucket_score)
    out["Suggested Action"] = out["Lead Tier"].apply(suggested_action)
    out["Tier Rank"] = out["Lead Tier"].map(TIER_ORDER)
    out = out.sort_values(["Tier Rank", "Lead Score", "Predicted Probability"], ascending=[True, False, False])
    return out.drop(columns=["Tier Rank"])


def load_input_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(DEFAULT_DATA_PATH)


st.set_page_config(page_title="Lead Scoring Dashboard", layout="wide")

st.title("Lead Scoring Dashboard")


with st.sidebar:
    st.header("Model Setup")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    score_button = st.button("Score Leads", type="primary")

model_path =str(DEFAULT_MODEL_PATH)
preprocessor_path = str(DEFAULT_PREPROCESSOR_PATH)

if score_button:
    model_file = Path(model_path)
    preprocessor_file = Path(preprocessor_path)

    if not model_file.exists():
        st.error(f"Model file not found: {model_file}")
        st.stop()
    if not preprocessor_file.exists():
        st.error(f"Preprocessor file not found: {preprocessor_file}")
        st.stop()

    try:
        df = load_input_data(uploaded_file)
        model, preprocessor = load_artifacts(str(model_file), str(preprocessor_file))
        scored = score_dataframe(df, model, preprocessor)

        hot_count = int((scored["Lead Tier"] == "Hot").sum())
        warm_count = int((scored["Lead Tier"] == "Warm").sum())
        cold_count = int((scored["Lead Tier"] == "Cold").sum())
        avg_score = float(scored["Lead Score"].mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Leads", len(scored))
        c2.metric("Hot Leads", hot_count)
        c3.metric("Warm Leads", warm_count)
        c4.metric("Average Lead Score", f"{avg_score:.1f}")

        st.subheader("Lead Tier Distribution")
        tier_counts = scored["Lead Tier"].value_counts().reindex(["Hot", "Warm", "Cold"], fill_value=0)
        st.bar_chart(tier_counts)

        st.subheader("Ranked Lead Table")
        display_cols = ["Lead ID", "Predicted Probability", "Lead Score", "Lead Tier", "Suggested Action"]
        st.dataframe(
            scored[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = scored.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download ranked leads as CSV",
            data=csv_bytes,
            file_name="ranked_leads.csv",
            mime="text/csv",
        )

        st.subheader("Top 10 Leads")
        st.dataframe(scored[display_cols].head(10), use_container_width=True, hide_index=True)

       

    except Exception as exc:
        st.error(f"Scoring failed: {exc}")
else:
    st.info("Upload a CSV or use the default dataset, then click 'Score Leads' to generate the ranked output.")
    
