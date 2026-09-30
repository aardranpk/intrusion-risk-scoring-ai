import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app.main import ModelBundle, app, get_model_bundle


class FakeModel:
    """Stands in for XGBoost so API tests run without a trained model or dataset."""

    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        return np.array([[1 - self.prob, self.prob]] * len(X))


FEATURES = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets"]


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_model(prob):
    app.dependency_overrides[get_model_bundle] = lambda: ModelBundle(FakeModel(prob), FEATURES)


def use_no_model():
    app.dependency_overrides[get_model_bundle] = lambda: None


def test_health_reports_model_status(client):
    use_no_model()
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
    use_model(0.5)
    assert client.get("/health").json()["model_loaded"] is True


def test_score_endpoint_returns_risk_and_severity(client):
    r = client.post("/score", json={"probability": 0.75})
    assert r.status_code == 200
    assert r.json() == {"risk_score": 75, "severity": "High", "model_probability": 0.75}


@pytest.mark.parametrize("bad", [-0.1, 1.5, "high"])
def test_score_endpoint_rejects_invalid_probability(client, bad):
    assert client.post("/score", json={"probability": bad}).status_code == 422


def test_predict_uses_model_output(client):
    use_model(0.95)
    payload = {"features": {f: 1.0 for f in FEATURES}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["severity"] == "Critical"
    assert r.json()["risk_score"] == 95


def test_predict_reports_missing_features(client):
    use_model(0.5)
    r = client.post("/predict", json={"features": {"Flow Duration": 1.0}})
    assert r.status_code == 422
    assert r.json()["detail"]["count"] == 2


def test_predict_returns_503_when_model_not_trained(client):
    use_no_model()
    r = client.post("/predict", json={"features": {f: 1.0 for f in FEATURES}})
    assert r.status_code == 503
