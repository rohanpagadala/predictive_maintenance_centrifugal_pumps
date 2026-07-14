"""
Loads the persisted model artifacts exactly once and caches them for the
life of the process.

Both the FastAPI app and the Streamlit dashboard import `get_artifacts()`
rather than calling `model_persistence.load_models()` directly -- this is
the one place that decides "load once at startup, reuse everywhere" instead
of every request paying joblib's deserialization cost.
"""

from __future__ import annotations

import logging
import time

import model_persistence as mp
from app import config, monitoring
from app.utils import ArtifactLoadError, timed

logger = logging.getLogger(__name__)

_artifacts: mp.ModelArtifacts | None = None
_load_time_seconds: float | None = None
_loaded_at: float | None = None


def get_artifacts(force_reload: bool = False) -> mp.ModelArtifacts:
    """Return the cached `ModelArtifacts`, loading them from disk on first call.

    Raises `ArtifactLoadError` (never the raw `ModelPersistenceError`) so
    callers only need to catch one exception type from this module.
    """
    global _artifacts, _load_time_seconds, _loaded_at

    if _artifacts is not None and not force_reload:
        return _artifacts

    with timed(logger, "model artifact loading") as t:
        try:
            _artifacts = mp.load_models(config.MODELS_DIR)
        except mp.ModelPersistenceError as exc:
            logger.error("Could not load model artifacts from %s", config.MODELS_DIR)
            raise ArtifactLoadError(str(exc)) from exc

    _load_time_seconds = t["elapsed_ms"] / 1000
    _loaded_at = time.time()
    monitoring.record_model_load(t["elapsed_ms"])
    logger.info(
        "Artifacts ready: classifier=%s regressor=%s n_features=%d (%.1f ms)",
        _artifacts.classifier_name, _artifacts.regressor_name,
        len(_artifacts.feature_list), t["elapsed_ms"],
    )
    return _artifacts


def get_load_time_seconds() -> float | None:
    """Wall-clock seconds the last artifact load took -- exposed for /health."""
    return _load_time_seconds


def get_uptime_seconds() -> float:
    """Seconds since artifacts were first loaded, or 0 if not loaded yet."""
    return time.time() - _loaded_at if _loaded_at is not None else 0.0


def is_loaded() -> bool:
    return _artifacts is not None
