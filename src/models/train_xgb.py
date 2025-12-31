from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump

from xgboost import XGBClassifier

from src.data.load_cicids import load_data

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Label"

def make_binary_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().apply(lambda x: 0 if x == "benign" else 1)

def train():
    df = load_data()

    X = df.drop(columns=[TARGET_COL])
    y = make_binary_label(df[TARGET_COL])

    # numeric-only baseline for speed and stability
    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # scale_pos_weight helps with imbalance (attacks are rare)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / max(pos, 1))

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report (XGBoost) ===")
    print(classification_report(y_test, preds, digits=4))

    auc = roc_auc_score(y_test, probs)
    print(f"ROC-AUC: {auc:.4f}")

    dump(model, ARTIFACTS_DIR / "xgb_model.joblib")
    dump(list(X.columns), ARTIFACTS_DIR / "feature_columns.joblib")
    print(f"\nSaved model to: {ARTIFACTS_DIR / 'xgb_model.joblib'}")

if __name__ == "__main__":
    train()
