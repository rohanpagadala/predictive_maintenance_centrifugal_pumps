from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

import model_registry as mr
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app import config, model_loader, monitoring, predict as predict_pipeline, preprocessing, security
from app.schemas import ErrorResponse, ExplanationResponse, HealthResponse, PredictionRequest, PredictionResponse
from app.utils import ArtifactLoadError, InferenceError, InputValidationError, setup_logging

setup_logging()
logger = logging.getLogger("pdm.api")

_start_time = time.time()
_PROTECTED = [Depends(security.require_api_key)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading model artifacts...")
    try:
        model_loader.get_artifacts()
    except ArtifactLoadError:
        logger.exception("Startup failed: could not load model artifacts.")
        raise
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan,
)

app.state.limiter = security.limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    monitoring.record_request(success=response.status_code < 400, latency_ms=elapsed_ms)
    return response


@app.exception_handler(InputValidationError)
async def handle_validation_error(request: Request, exc: InputValidationError):
    logger.warning("Input validation error on %s: %s", request.url.path, exc)
    return _error_response(422, "input_validation_error", str(exc), sensitive=False)


@app.exception_handler(InferenceError)
async def handle_inference_error(request: Request, exc: InferenceError):
    logger.error("Inference error on %s: %s", request.url.path, exc)
    return _error_response(500, "inference_error", str(exc), sensitive=True)


@app.exception_handler(ArtifactLoadError)
async def handle_artifact_error(request: Request, exc: ArtifactLoadError):
    logger.error("Artifact load error on %s: %s", request.url.path, exc)
    return _error_response(503, "model_unavailable", str(exc), sensitive=True)


def _unexpected_error_detail(exc: Exception) -> str:
    return f"Unexpected error: {exc}" if config.VERBOSE_ERRORS else "An internal error occurred. Check server logs for details."


def _error_response(status_code: int, error: str, detail: str, sensitive: bool = True):
    from fastapi.responses import JSONResponse
    if sensitive and not config.VERBOSE_ERRORS:
        detail = "An internal error occurred. Check server logs for details."
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error, detail=detail).model_dump(),
    )


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": config.APP_NAME, "version": config.APP_VERSION}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    loaded = model_loader.is_loaded()
    artifacts = model_loader.get_artifacts() if loaded else None
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        classifier_name=artifacts.classifier_name if artifacts else None,
        regressor_name=artifacts.regressor_name if artifacts else None,
        n_features=len(artifacts.feature_list) if artifacts else None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics", tags=["Health"])
async def metrics():
    return monitoring.snapshot()


@app.get("/metrics/prometheus", tags=["Health"])
async def metrics_prometheus():
    drift_report = model_loader.get_drift_monitor().compute_drift_report() if model_loader.is_loaded() else None
    body, content_type = monitoring.prometheus_exposition(drift_report)
    return Response(content=body, media_type=content_type)


@app.get("/drift", tags=["Monitoring"])
async def drift():
    return model_loader.get_drift_monitor().compute_drift_report()


@app.get("/audit/recent", tags=["Monitoring"], dependencies=_PROTECTED)
async def audit_recent(limit: int = 100):
    limit = max(1, min(limit, 1000))
    return {"records": model_loader.get_audit_logger().get_recent(limit)}


@app.get("/audit/asset/{asset_id}", tags=["Monitoring"], dependencies=_PROTECTED)
async def audit_by_asset(asset_id: str, limit: int = 100):
    limit = max(1, min(limit, 1000))
    return {"records": model_loader.get_audit_logger().get_by_asset(asset_id, limit)}


@app.get("/models", tags=["Model Registry"], dependencies=_PROTECTED)
async def list_models():
    return {"versions": mr.list_versions(config.MODELS_DIR)}


@app.post("/models/{version}/promote", tags=["Model Registry"], dependencies=_PROTECTED)
async def promote_model(version: str):
    try:
        mr.promote_to_stable(config.MODELS_DIR, version)
    except mr.ModelRegistryError as exc:
        raise HTTPException(status_code=404, detail=_unexpected_error_detail(exc)) from exc
    model_loader.get_artifacts(force_reload=True)
    return {"status": "promoted", "active_version": model_loader.get_active_version()}


@app.post("/models/rollback", tags=["Model Registry"], dependencies=_PROTECTED)
async def rollback_model():
    try:
        previous = mr.rollback_to_previous(config.MODELS_DIR)
    except mr.ModelRegistryError as exc:
        raise HTTPException(status_code=409, detail=_unexpected_error_detail(exc)) from exc
    model_loader.get_artifacts(force_reload=True)
    return {"status": "rolled_back", "active_version": previous}


@app.get("/explain/summary", tags=["Prediction"], dependencies=_PROTECTED)
async def explain_summary_endpoint():
    try:
        return await run_in_threadpool(model_loader.get_summary_plots)
    except ArtifactLoadError as exc:
        raise HTTPException(status_code=503, detail=_unexpected_error_detail(exc)) from exc


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], dependencies=_PROTECTED)
@security.limiter.limit(config.RATE_LIMIT_PREDICT)
async def predict_endpoint(request: Request, body: PredictionRequest):
    try:
        return await run_in_threadpool(predict_pipeline.predict, body)
    except InputValidationError:
        raise
    except InferenceError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /predict")
        raise HTTPException(status_code=500, detail=_unexpected_error_detail(exc)) from exc


@app.post("/explain", response_model=ExplanationResponse, tags=["Prediction"], dependencies=_PROTECTED)
@security.limiter.limit(config.RATE_LIMIT_PREDICT)
async def explain_endpoint(request: Request, body: PredictionRequest):
    try:
        return await run_in_threadpool(predict_pipeline.explain, body)
    except InputValidationError:
        raise
    except InferenceError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /explain")
        raise HTTPException(status_code=500, detail=_unexpected_error_detail(exc)) from exc


@app.post("/predict-csv", tags=["Prediction"], dependencies=_PROTECTED)
@security.limiter.limit(config.RATE_LIMIT_PREDICT)
async def predict_csv_endpoint(request: Request, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")

    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({int(content_length):,} bytes); the limit is {config.MAX_UPLOAD_BYTES:,} bytes.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw_bytes):,} bytes); the limit is {config.MAX_UPLOAD_BYTES:,} bytes.",
        )
    try:
        input_df = preprocessing.dataframe_from_csv_bytes(raw_bytes)
        result_df: pd.DataFrame = await run_in_threadpool(predict_pipeline.predict_batch, input_df)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=_unexpected_error_detail(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /predict-csv")
        raise HTTPException(status_code=500, detail=_unexpected_error_detail(exc)) from exc

    buffer = io.StringIO()
    result_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
