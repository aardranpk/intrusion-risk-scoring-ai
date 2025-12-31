from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

from src.data.load_cicids import load_data

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Label"

def make_binary_label(series: pd.Series) -> pd.Series:
    # dataset uses "Benign" for normal traffic
    # everything else is considered "attack" for baseline binary detection
    return series.astype(str).str.lower().apply(lambda x: 0 if x == "benign" else 1)

def train():
    df = load_data()

    # Keep only numeric features (safe baseline)
    X = df.drop(columns=[TARGET_COL])
    y = make_binary_label(df[TARGET_COL])

    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, digits=4))

    try:
        auc = roc_auc_score(y_test, probs)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        print("ROC-AUC could not be computed:", e)

    dump(pipe, ARTIFACTS_DIR / "baseline_logreg.joblib")
    print(f"\nSaved model to: {ARTIFACTS_DIR / 'baseline_logreg.joblib'}")

if __name__ == "__main__":
    train()
