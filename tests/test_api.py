from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from api import main


class FakeModel:
    def predict(self, frame):
        assert not frame.empty
        return np.array([1])

    def predict_proba(self, frame):
        assert not frame.empty
        return np.array([[0.2, 0.8]])


def test_health_endpoint_reports_model_state(monkeypatch, tmp_path):
    missing = tmp_path / "missing.joblib"
    monkeypatch.setenv("MODEL_PATH", str(missing))
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_available"] is False


def test_prediction_endpoint(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda: FakeModel())
    client = TestClient(main.app)

    response = client.post(
        "/predict",
        json={"features": {"TSH": 8.2, "FTI": 55.0, "sex": "F"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "hypothyroid"
    assert body["probability_hypothyroid"] == 0.8
    assert "não constitui diagnóstico" in body["disclaimer"]


def test_prediction_requires_features():
    client = TestClient(main.app)

    response = client.post("/predict", json={"features": {}})

    assert response.status_code == 422
