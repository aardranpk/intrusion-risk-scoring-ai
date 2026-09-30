"""
REST API for the Intrusion Risk Scoring System.

Endpoints
---------
GET  /health   -> service status and whether a trained model is loaded
POST /score    -> convert a model probability (0..1) into a risk score + severity
POST /predict  -> score one network flow (feature dict) with the trained XGBoost model

Run locally:
    uvicorn src.app.main:app --reload
Interactive docs:
    http://127.0.0.1:8000/docs
"""
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.risk_scoring import make_risk_output

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "xgb_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.joblib"

app = FastAPI(
    title="Intrusion Risk Scoring API",
    version="1.0.0",
    description="Turns intrusion-detection model output into risk scores and severity levels.",
)


# ---------- Schemas ----------
class ScoreRequest(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0, description="Model probability of attack (0..1)")


class FlowRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Numeric flow features keyed by column name")


class RiskResponse(BaseModel):
    risk_score: int
    severity: str
    model_probability: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------- Model loading ----------
class ModelBundle:
    """Holds the trained model and the feature order it expects."""

    def __init__(self, model, feature_columns: List[str]):
        self.model = model
        self.feature_columns = feature_columns


@lru_cache(maxsize=1)
def _load_bundle() -> Optional[ModelBundle]:
    if not (MODEL_PATH.exists() and FEATURES_PATH.exists()):
        return None
    from joblib import load  # imported lazily so /score works without ML deps

    return ModelBundle(load(MODEL_PATH), load(FEATURES_PATH))


def get_model_bundle() -> Optional[ModelBundle]:
    """FastAPI dependency; overridden in tests with a fake model."""
    return _load_bundle()


# ---------- Routes ----------
@app.get("/health", response_model=HealthResponse)
def health(bundle: Optional[ModelBundle] = Depends(get_model_bundle)):
    return HealthResponse(status="ok", model_loaded=bundle is not None)


@app.post("/score", response_model=RiskResponse)
def score(req: ScoreRequest):
    out = make_risk_output(req.probability)
    return RiskResponse(**out.__dict__)


@app.post("/predict", response_model=RiskResponse)
def predict(req: FlowRequest, bundle: Optional[ModelBundle] = Depends(get_model_bundle)):
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python -m src.models.train_xgb` first.",
        )

    missing = [c for c in bundle.feature_columns if c not in req.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"error": "Missing required features", "missing": missing[:10], "count": len(missing)},
        )

    import pandas as pd

    row = pd.DataFrame([[req.features[c] for c in bundle.feature_columns]], columns=bundle.feature_columns)
    prob = float(bundle.model.predict_proba(row)[0, 1])
    out = make_risk_output(prob)
    return RiskResponse(**out.__dict__)
