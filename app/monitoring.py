from __future__ import annotations

import statistics
import threading
import time
from collections import deque

import psutil
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

_LOCK = threading.Lock()
_MAX_LATENCY_SAMPLES = 500
_MAX_CONFIDENCE_SAMPLES = 500

_state = {
    "total_requests": 0,
    "failed_requests": 0,
    "prediction_count": 0,
    "latencies_ms": deque(maxlen=_MAX_LATENCY_SAMPLES),
    "confidences": deque(maxlen=_MAX_CONFIDENCE_SAMPLES),
    "model_load_time_ms": None,
    "started_at": time.time(),
}

REGISTRY = CollectorRegistry()
_process = psutil.Process()

_requests_total = Counter("pdm_requests_total", "Total HTTP requests handled.", registry=REGISTRY)
_requests_failed_total = Counter("pdm_requests_failed_total", "HTTP requests that returned an error status.", registry=REGISTRY)
_request_latency_seconds = Histogram(
    "pdm_request_latency_seconds", "HTTP request latency in seconds.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
    registry=REGISTRY,
)
_predictions_total = Counter("pdm_predictions_total", "Total individual predictions made (single + batch rows).", registry=REGISTRY)
_prediction_confidence = Histogram(
    "pdm_prediction_confidence", "Classifier confidence (max class probability) per prediction.",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)
_model_load_time_seconds = Gauge("pdm_model_load_time_seconds", "Wall-clock time of the last model artifact load.", registry=REGISTRY)
_cpu_usage_percent = Gauge("pdm_cpu_usage_percent", "Process CPU usage percent, sampled at scrape time.", registry=REGISTRY)
_memory_usage_mb = Gauge("pdm_memory_usage_mb", "Process resident memory in MB, sampled at scrape time.", registry=REGISTRY)
_drift_status_code = Gauge(
    "pdm_drift_status", "Overall data drift status: 0=none, 1=moderate, 2=significant.", registry=REGISTRY
)
_drift_psi_by_feature = Gauge(
    "pdm_drift_psi", "Population Stability Index per monitored feature.", ["feature"], registry=REGISTRY
)

_DRIFT_STATUS_CODES = {"none": 0, "moderate": 1, "significant": 2, "unavailable": -1}


def record_request(success: bool, latency_ms: float) -> None:
    with _LOCK:
        _state["total_requests"] += 1
        if not success:
            _state["failed_requests"] += 1
        _state["latencies_ms"].append(latency_ms)
    _requests_total.inc()
    if not success:
        _requests_failed_total.inc()
    _request_latency_seconds.observe(latency_ms / 1000)


def record_prediction(confidence: float | None = None) -> None:
    with _LOCK:
        _state["prediction_count"] += 1
        if confidence is not None:
            _state["confidences"].append(confidence)
    _predictions_total.inc()
    if confidence is not None:
        _prediction_confidence.observe(confidence)


def record_model_load(latency_ms: float) -> None:
    with _LOCK:
        _state["model_load_time_ms"] = latency_ms
    _model_load_time_seconds.set(latency_ms / 1000)


def snapshot() -> dict:
    with _LOCK:
        latencies = list(_state["latencies_ms"])
        confidences = list(_state["confidences"])
        return {
            "total_requests": _state["total_requests"],
            "failed_requests": _state["failed_requests"],
            "failure_rate": round(_state["failed_requests"] / _state["total_requests"], 4)
            if _state["total_requests"] else 0.0,
            "prediction_count": _state["prediction_count"],
            "model_load_time_ms": _state["model_load_time_ms"],
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
            "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2)
            if len(latencies) >= 20 else None,
            "avg_confidence_score": round(statistics.mean(confidences), 4) if confidences else None,
            "uptime_seconds": round(time.time() - _state["started_at"], 1),
        }


def prometheus_exposition(drift_report: dict | None = None) -> tuple[bytes, str]:
    _cpu_usage_percent.set(_process.cpu_percent(interval=None))
    _memory_usage_mb.set(_process.memory_info().rss / (1024 * 1024))

    if drift_report and drift_report.get("status") == "ok":
        _drift_status_code.set(_DRIFT_STATUS_CODES.get(drift_report["overall_drift_status"], -1))
        for feature_report in drift_report.get("covariate_drift", []):
            _drift_psi_by_feature.labels(feature=feature_report["feature"]).set(feature_report["psi"])
    else:
        _drift_status_code.set(_DRIFT_STATUS_CODES["unavailable"])

    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
