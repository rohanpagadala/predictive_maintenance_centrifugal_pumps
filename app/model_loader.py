from __future__ import annotations

import logging
import time

import joblib
import pandas as pd

import model_persistence as mp
import model_registry as mr
from app import config, drift_detection as dd, explainability as xai, monitoring
from app.audit_log import AuditLogger
from app.utils import ArtifactLoadError, timed

logger = logging.getLogger(__name__)

_artifacts: mp.ModelArtifacts | None = None
_active_version: str | None = None
_load_time_seconds: float | None = None
_loaded_at: float | None = None
_drift_monitor: dd.DriftMonitor | None = None
_explainers: xai.Explainers | None = None
_background_sample: pd.DataFrame | None = None
_summary_plots: dict | None = None
_audit_logger: AuditLogger | None = None


def get_artifacts(force_reload: bool = False, version: str | None = None) -> mp.ModelArtifacts:
    global _artifacts, _active_version, _load_time_seconds, _loaded_at, _drift_monitor, _explainers, _background_sample, _summary_plots

    if _artifacts is not None and not force_reload and version is None:
        return _artifacts

    with timed(logger, "model artifact loading") as t:
        try:
            _artifacts = mr.load_models_from_registry(config.MODELS_DIR, version=version)
            _active_version = _artifacts.metadata.get("model_version", version or "unknown")
        except (mp.ModelPersistenceError, mr.ModelRegistryError) as exc:
            logger.error("Could not load model artifacts from %s", config.MODELS_DIR)
            raise ArtifactLoadError(str(exc)) from exc

    _load_time_seconds = t["elapsed_ms"] / 1000
    _loaded_at = time.time()
    monitoring.record_model_load(t["elapsed_ms"])
    logger.info(
        "Artifacts ready: version=%s classifier=%s regressor=%s n_features=%d (%.1f ms)",
        _active_version, _artifacts.classifier_name, _artifacts.regressor_name,
        len(_artifacts.feature_list), t["elapsed_ms"],
    )

    reference_path = mr.get_version_dir(config.MODELS_DIR, _active_version) / dd.REFERENCE_FILENAME
    reference = dd.load_reference(reference_path) if reference_path.exists() else None
    if reference is None:
        logger.warning("No drift reference found at %s -- /drift will report 'unavailable' until one is built.", reference_path)
    _drift_monitor = dd.DriftMonitor(reference)
    _explainers = None

    background_path = mr.get_version_dir(config.MODELS_DIR, _active_version) / xai.BACKGROUND_SAMPLE_FILENAME
    _background_sample = joblib.load(background_path) if background_path.exists() else None
    _summary_plots = None

    return _artifacts


def get_active_version() -> str | None:
    return _active_version


def get_drift_monitor() -> dd.DriftMonitor:
    if _drift_monitor is None:
        get_artifacts()
    return _drift_monitor


def get_explainers() -> xai.Explainers:
    global _explainers
    if _explainers is None:
        artifacts = get_artifacts()
        with timed(logger, "SHAP explainer construction"):
            _explainers = xai.build_explainers(artifacts.classifier, artifacts.regressor)
    return _explainers


def get_summary_plots() -> dict:
    global _summary_plots
    if _summary_plots is None:
        if _background_sample is None:
            raise ArtifactLoadError(
                f"No SHAP background sample found for model version {_active_version} "
                f"-- summary plots unavailable until one is built (see app/explainability.py)."
            )
        explainers = get_explainers()
        with timed(logger, "SHAP summary plot generation"):
            _summary_plots = xai.generate_summary_plots(explainers, _background_sample, _artifacts.feature_list)
    return _summary_plots


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(config.LOGS_DIR)
    return _audit_logger


def get_load_time_seconds() -> float | None:
    return _load_time_seconds


def get_uptime_seconds() -> float:
    return time.time() - _loaded_at if _loaded_at is not None else 0.0


def is_loaded() -> bool:
    return _artifacts is not None
