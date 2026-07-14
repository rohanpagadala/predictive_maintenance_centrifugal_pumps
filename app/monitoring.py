"""
In-process monitoring counters, exposed via `GET /metrics`.

Deliberately simple -- in-memory, single-process counters, not a full
Prometheus/OpenTelemetry integration. That's the right scope for this
project; every other module only calls the three `record_*()` functions
below, so swapping this out for a real metrics backend later touches only
this file.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque

_LOCK = threading.Lock()
_MAX_LATENCY_SAMPLES = 500

_state = {
    "total_requests": 0,
    "failed_requests": 0,
    "prediction_count": 0,
    "latencies_ms": deque(maxlen=_MAX_LATENCY_SAMPLES),
    "model_load_time_ms": None,
    "started_at": time.time(),
}


def record_request(success: bool, latency_ms: float) -> None:
    """Called once per HTTP request by the API's logging middleware."""
    with _LOCK:
        _state["total_requests"] += 1
        if not success:
            _state["failed_requests"] += 1
        _state["latencies_ms"].append(latency_ms)


def record_prediction() -> None:
    """Called once per individual prediction made (single or per CSV row)."""
    with _LOCK:
        _state["prediction_count"] += 1


def record_model_load(latency_ms: float) -> None:
    """Called once by app.model_loader after artifacts are loaded."""
    with _LOCK:
        _state["model_load_time_ms"] = latency_ms


def snapshot() -> dict:
    """A point-in-time summary for the /metrics endpoint."""
    with _LOCK:
        latencies = list(_state["latencies_ms"])
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
            "uptime_seconds": round(time.time() - _state["started_at"], 1),
        }
