"""API cloud-ready para inferência do modelo otimizado.

O notebook continua responsável pelo treinamento. Esta API carrega o artefato
Joblib produzido pelo notebook e oferece endpoints de saúde, predição e métricas.
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field


class JsonFormatter(logging.Formatter):
    """Formata logs operacionais em JSON para coleta por serviços de nuvem."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("tech_challenge_api")
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logger.propagate = False

REQUESTS = Counter(
    "tech_challenge_prediction_requests_total",
    "Total de requisições de predição.",
    ["status"],
)
LATENCY = Histogram(
    "tech_challenge_prediction_latency_seconds",
    "Latência das predições em segundos.",
)


class PredictionRequest(BaseModel):
    """Valores de uma observação nas mesmas colunas usadas no treinamento."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ...,
        min_length=1,
        description="Mapa coluna-valor com as variáveis clínicas do modelo.",
    )


class PredictionResponse(BaseModel):
    predicted_class: str
    probability_negative: float
    probability_hypothyroid: float
    disclaimer: str


def model_path() -> Path:
    return Path(
        os.getenv("MODEL_PATH", "models/random_forest_otimizado.joblib")
    )


@lru_cache(maxsize=1)
def load_model() -> Any:
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {path}. Execute o notebook de treinamento."
        )
    logger.info("Carregando modelo de %s", path)
    return joblib.load(path)


app = FastAPI(
    title="Tech Challenge - Hipotireoidismo",
    version="1.0.0",
    description=(
        "API acadêmica de apoio à triagem. Não substitui diagnóstico médico."
    ),
)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health() -> dict[str, Any]:
    path = model_path()
    return {
        "status": "ok",
        "model_available": path.exists(),
        "model_path": str(path),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    started = time.perf_counter()
    try:
        model = load_model()
        frame = pd.DataFrame([payload.features])
        prediction = int(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        response = PredictionResponse(
            predicted_class="hypothyroid" if prediction == 1 else "negative",
            probability_negative=float(probabilities[0]),
            probability_hypothyroid=float(probabilities[1]),
            disclaimer=(
                "Resultado acadêmico de apoio à triagem; não constitui diagnóstico."
            ),
        )
        REQUESTS.labels(status="success").inc()
        logger.info(
            "Predição concluída classe=%s latencia=%.4f",
            response.predicted_class,
            time.perf_counter() - started,
        )
        return response
    except FileNotFoundError as exc:
        REQUESTS.labels(status="model_unavailable").inc()
        logger.error("Modelo indisponível: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        REQUESTS.labels(status="error").inc()
        logger.exception("Falha durante a predição")
        raise HTTPException(status_code=422, detail="Falha ao processar os dados.") from exc
    finally:
        LATENCY.observe(time.perf_counter() - started)
