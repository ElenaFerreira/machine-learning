"""FastAPI app — endpoints and lifespan (S5: monitoring + drift detection)."""
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.config import (
    API_TITLE, API_VERSION, API_DESCRIPTION,
    MODEL_PATH, MANIFEST_PATH,
    PSI_OK_THRESHOLD, PSI_WARNING_THRESHOLD,
)
from app.model import ModelService
from app.schemas import (
    ClientFeatures, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse,
    MetricsResponse, DriftCheckResponse,
)
from app.monitoring import (
    log_prediction, read_recent_logs,
    compute_drift, status_from_max_psi, get_baseline,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE at startup, share it via app.state."""
    app.state.model_service = ModelService(MODEL_PATH, MANIFEST_PATH)
    print(f"✅ Model loaded: {app.state.model_service.manifest['model_name']}")
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect to Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    """Liveness probe."""
    model_loaded = hasattr(app.state, "model_service") and app.state.model_service is not None
    return HealthResponse(
        status="ok" if model_loaded else "error",
        model_loaded=model_loaded,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["service"])
def model_info():
    """Return manifest content (model metadata)."""
    return ModelInfoResponse(**app.state.model_service.manifest)


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict(client: ClientFeatures):
    """Predict churn for ONE client. Logs the prediction for monitoring."""
    try:
        features = client.model_dump()
        result = app.state.model_service.predict_one(features)
        log_prediction(features, result)  # ← NEW: log every prediction
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["predict"])
def predict_batch(req: BatchPredictionRequest):
    """Predict churn for N clients in one shot. Logs each prediction."""
    try:
        features_list = [c.model_dump() for c in req.clients]
        results = app.state.model_service.predict_batch(features_list)
        for feats, res in zip(features_list, results):
            log_prediction(feats, res)  # ← NEW: log each prediction in batch
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
def metrics():
    """Recent traffic stats from the JSONL log."""
    logs = read_recent_logs(1000)
    if not logs:
        return MetricsResponse(
            total_predictions=0,
            churn_rate_predicted=0.0,
            avg_churn_probability=0.0,
        )
    probas = [entry["prediction"]["churn_probability"] for entry in logs]
    predicted = [entry["prediction"]["churn_predicted"] for entry in logs]
    return MetricsResponse(
        total_predictions=len(logs),
        churn_rate_predicted=sum(predicted) / len(predicted),
        avg_churn_probability=sum(probas) / len(probas),
    )


@app.get("/drift-check", response_model=DriftCheckResponse, tags=["monitoring"])
def drift_check(min_samples: int = 50):
    """Compare recent input distributions to the training baseline (PSI)."""
    logs = read_recent_logs(1000)
    thresholds = {"ok": PSI_OK_THRESHOLD, "warning": PSI_WARNING_THRESHOLD}

    if len(logs) < min_samples:
        return DriftCheckResponse(
            status="insufficient_data",
            n_predictions_analyzed=len(logs),
            drift_scores={},
            thresholds=thresholds,
        )

    recent_df = pd.DataFrame([entry["features"] for entry in logs])
    baseline = get_baseline()
    scores = compute_drift(baseline, recent_df)

    return DriftCheckResponse(
        status=status_from_max_psi(scores),
        n_predictions_analyzed=len(logs),
        drift_scores=scores,
        thresholds=thresholds,
    )