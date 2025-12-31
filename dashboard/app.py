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

st.sidebar.header("Select a traffic flow")
row_idx = st.sidebar.number_input("Row index", min_value=0, max_value=len(X)-1, value=0, step=1)

row = X.iloc[[row_idx]]
prob = float(model.predict_proba(row)[:, 1][0])
risk = make_risk_output(prob)

c1, c2, c3 = st.columns(3)
c1.metric("Model Probability", f"{risk.model_probability:.4f}")
c2.metric("Risk Score (0-100)", f"{risk.risk_score}")
c3.metric("Severity", risk.severity)

st.subheader("Top reasons (SHAP)")
exp = explain_one_row(row, top_k=7)

reason_df = pd.DataFrame(exp.top_features, columns=["feature", "contribution"])
reason_df["direction"] = reason_df["contribution"].apply(lambda v: "↑ risk" if v > 0 else "↓ risk")
st.dataframe(reason_df, use_container_width=True)

with st.expander("View raw feature values"):
    st.dataframe(row.T, use_container_width=True)
