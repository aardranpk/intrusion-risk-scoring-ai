from joblib import load
from src.data.load_cicids import load_data
from src.models.risk_scoring import make_risk_output

def main():
    df = load_data()
    X = df.drop(columns=["Label"]).select_dtypes(include=["number"])

    model = load("artifacts/xgb_model.joblib")

    # demo on first 5 rows
    probs = model.predict_proba(X.head(5))[:, 1]
    for i, p in enumerate(probs):
        out = make_risk_output(p)
        print(f"Row {i} -> prob={p:.4f} | risk={out.risk_score} | severity={out.severity}")

if __name__ == "__main__":
    main()
