"""
FastAPI application: the HTTP front door to `app.predict`.

Every endpoint is a thin wrapper around the inference pipeline in
`app/predict.py` -- no preprocessing, feature engineering, or model logic
lives here, only request/response handling, validation, logging, and error
mapping. That keeps the inference pipeline testable and reusable (e.g. from
the Streamlit dashboard) independent of the web framework.

Run locally:
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
Then open http://localhost:8000/docs for interactive Swagger UI.
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import config, model_loader, monitoring, predict as predict_pipeline, preprocessing
from app.schemas import ErrorResponse, HealthResponse, PredictionRequest, PredictionResponse
from app.utils import ArtifactLoadError, InferenceError, InputValidationError, setup_logging

setup_logging()
logger = logging.getLogger("pdm.api")

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup, not on the first request --
    so the process either comes up ready to serve or fails fast at boot."""
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

# Locked down to specific origins in production via PDM_ALLOWED_ORIGINS;
# permissive by default so the Streamlit dashboard (a different port/origin
# in local dev and in docker-compose) can call the API out of the box.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every request's method, path, status, and latency -- Phase 7's
    "log API requests" / "prediction latency" requirement in one place."""
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
    return _error_response(422, "input_validation_error", str(exc))


@app.exception_handler(InferenceError)
async def handle_inference_error(request: Request, exc: InferenceError):
    logger.error("Inference error on %s: %s", request.url.path, exc)
    return _error_response(500, "inference_error", str(exc))


@app.exception_handler(ArtifactLoadError)
async def handle_artifact_error(request: Request, exc: ArtifactLoadError):
    logger.error("Artifact load error on %s: %s", request.url.path, exc)
    return _error_response(503, "model_unavailable", str(exc))


def _error_response(status_code: int, error: str, detail: str):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error, detail=detail).model_dump(),
    )


@app.get("/", tags=["Health"])
async def root():
    """Liveness check -- confirms the process is up and responding."""
    return {"status": "ok", "service": config.APP_NAME, "version": config.APP_VERSION}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Readiness check -- confirms the model artifacts are actually loaded
    and reports basic model identity for a quick sanity check in production."""
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
    """Monitoring snapshot: request counts, failure rate, latency
    percentiles, prediction count, and model load time. Polled by an
    external monitor (or just curled manually) -- not a Prometheus exposition
    format, deliberately kept simple JSON for this project's scope."""
    return monitoring.snapshot()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_endpoint(request: PredictionRequest):
    """Score a single pump's reading history (1+ chronological readings).

    Runs the CPU-bound inference pipeline in a threadpool so it doesn't
    block the event loop under concurrent load.
    """
    try:
        return await run_in_threadpool(predict_pipeline.predict, request)
    except InputValidationError:
        raise
    except InferenceError:
        raise
    except Exception as exc:  # noqa: BLE001 -- last-resort catch-all, logged and mapped to 500
        logger.exception("Unexpected error in /predict")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@app.post("/predict-csv", tags=["Prediction"])
async def predict_csv_endpoint(file: UploadFile = File(...)):
    """Score every row of an uploaded CSV and return a downloadable CSV of
    predictions. Each row is treated as an independent reading unless rows
    share an Asset_ID, in which case they're used as that pump's chronological
    history and every row still gets its own prediction.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")

    raw_bytes = await file.read()
    try:
        input_df = preprocessing.dataframe_from_csv_bytes(raw_bytes)
        result_df: pd.DataFrame = await run_in_threadpool(predict_pipeline.predict_batch, input_df)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /predict-csv")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    buffer = io.StringIO()
    result_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
