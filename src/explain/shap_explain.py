from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
from joblib import load

@dataclass(frozen=True)
class Explanation:
    top_features: List[Tuple[str, float]]  # (feature_name, contribution)

def load_artifacts():
    model = load("artifacts/xgb_model.joblib")
    feature_cols = load("artifacts/feature_columns.joblib")
    return model, feature_cols

def explain_one_row(X_row: pd.DataFrame, top_k: int = 5) -> Explanation:
    """
    Explain a single row prediction using SHAP values for the XGBoost model.
    X_row must be a 1-row dataframe with numeric columns.
    """
    model, feature_cols = load_artifacts()

    # Ensure column order matches training
    X_row = X_row[feature_cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)

    # shap_values shape: (1, n_features)
    vals = np.array(shap_values).reshape(-1)
    abs_vals = np.abs(vals)

    idx = np.argsort(abs_vals)[::-1][:top_k]
    top = [(feature_cols[i], float(vals[i])) for i in idx]

    return Explanation(top_features=top)
