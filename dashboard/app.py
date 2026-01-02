import streamlit as st
import pandas as pd
from joblib import load

from src.data.load_cicids import load_data
from src.models.risk_scoring import make_risk_output
from src.explain.shap_explain import explain_one_row

st.set_page_config(page_title="Intrusion Risk Scoring", layout="wide")

@st.cache_resource
def load_model():
    return load("artifacts/xgb_model.joblib")

@st.cache_data
def load_df():
    df = load_data()
    X = df.drop(columns=["Label"]).select_dtypes(include=["number"])
    return df, X

st.title("Explainable Intrusion Risk Scoring (CICIDS)")

model = load_model()
df, X = load_df()

# ---- Enhancement controls ----
st.sidebar.header("Triage Controls")
top_n = st.sidebar.slider("Top N risky flows", min_value=10, max_value=200, value=30, step=10)
threshold = st.sidebar.slider("Alert threshold (probability)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

# Compute probabilities for top-N table (cache-friendly)
@st.cache_data
def compute_probs(_X: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(_X)[:, 1], index=_X.index)

probs_all = compute_probs(X)

# Top risky flows table
top_idx = probs_all.sort_values(ascending=False).head(top_n).index
top_table = pd.DataFrame({
    "row_index": top_idx,
    "probability": probs_all.loc[top_idx].values,
})
top_table["risk_score"] = top_table["probability"].apply(lambda p: make_risk_output(float(p)).risk_score)
top_table["severity"] = top_table["probability"].apply(lambda p: make_risk_output(float(p)).severity)
top_table["alert"] = top_table["probability"].apply(lambda p: "YES" if float(p) >= threshold else "NO")

st.subheader("🔥 Top Risky Flows (Triage View)")
st.dataframe(top_table, use_container_width=True)

# Downloadable report
csv_bytes = top_table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download triage report (CSV)",
    data=csv_bytes,
    file_name="triage_report.csv",
    mime="text/csv",
)

st.divider()

# Pick a row to explain (either manual or from top list)
st.sidebar.header("Explain a Flow")
mode = st.sidebar.radio("Select mode", ["Manual index", "Pick from Top Risky Flows"])

if mode == "Manual index":
    row_idx = st.sidebar.number_input("Row index", min_value=0, max_value=len(X)-1, value=0, step=1)
else:
    picked = st.sidebar.selectbox("Choose a row from top risky flows", top_table["row_index"].tolist())
    row_idx = int(picked)

row = X.iloc[[row_idx]]
prob = float(model.predict_proba(row)[:, 1][0])
risk = make_risk_output(prob)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model Probability", f"{risk.model_probability:.4f}")
c2.metric("Risk Score (0-100)", f"{risk.risk_score}")
c3.metric("Severity", risk.severity)
c4.metric("Alert (thresholded)", "YES" if prob >= threshold else "NO")

st.subheader("Top reasons (SHAP)")
exp = explain_one_row(row, top_k=7)

reason_df = pd.DataFrame(exp.top_features, columns=["feature", "contribution"])
reason_df["direction"] = reason_df["contribution"].apply(lambda v: "↑ risk" if v > 0 else "↓ risk")
st.dataframe(reason_df, use_container_width=True)

with st.expander("View raw feature values"):
    st.dataframe(row.T, use_container_width=True)
