from joblib import load
from src.data.load_cicids import load_data
from src.models.risk_scoring import make_risk_output
from src.explain.shap_explain import explain_one_row

def main():
    df = load_data()
    X = df.drop(columns=["Label"]).select_dtypes(include=["number"])

    model = load("artifacts/xgb_model.joblib")
    probs = model.predict_proba(X.head(1))[:, 1]
    out = make_risk_output(probs[0])

    print(f"prob={probs[0]:.4f} | risk={out.risk_score} | severity={out.severity}")

    exp = explain_one_row(X.head(1), top_k=7)
    print("\nTop reasons (SHAP contributions):")
    for name, val in exp.top_features:
        direction = "↑ risk" if val > 0 else "↓ risk"
        print(f"- {name}: {val:+.4f} ({direction})")

if __name__ == "__main__":
    main()
