from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

MONITORED_NUMERIC_COLS = ["Vibration_mm_s", "Temperature_C", "Pressure_psi", "Flow_Rate_m3_h"]
MONITORED_CATEGORICAL_COLS = ["Machine_Model", "Location"]
REFERENCE_FILENAME = "drift_reference.joblib"

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25
N_HISTOGRAM_BINS = 5
DEFAULT_MIN_LIVE_SAMPLES = 50
DEFAULT_BUFFER_SIZE = 500


@dataclass
class DriftReference:

    numeric: dict[str, dict[str, Any]]
    categorical: dict[str, dict[str, Any]]
    prediction_reference: dict[str, Any]
    built_at_utc: str
    n_reference_rows: int


def _psi(reference_freq: np.ndarray, live_freq: np.ndarray) -> float:
    eps = 1e-6
    ref = np.clip(reference_freq, eps, None)
    live = np.clip(live_freq, eps, None)
    return float(np.sum((live - ref) * np.log(live / ref)))


def _psi_band(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "none"


def build_reference_from_raw_data(
    raw_df: pd.DataFrame,
    predict_fn=None,
    engineer_fn=None,
    sample_size: int = 2000,
    random_state: int = 42,
) -> DriftReference:
    rng = np.random.RandomState(random_state)
    sample_df = raw_df.sample(n=min(sample_size, len(raw_df)), random_state=rng)

    numeric_ref: dict[str, dict[str, Any]] = {}
    for col in MONITORED_NUMERIC_COLS:
        if col not in sample_df.columns:
            continue
        values = sample_df[col].dropna().to_numpy()
        if len(values) == 0:
            logger.warning("Skipping drift reference for %r: no non-null sampled values.", col)
            continue
        bin_edges = np.unique(np.quantile(values, np.linspace(0, 1, N_HISTOGRAM_BINS + 1)))
        if len(bin_edges) < 2:
            bin_edges = np.array([values.min(), values.max() + 1e-9])
        counts, _ = np.histogram(values, bins=bin_edges)
        freq = counts / counts.sum()
        numeric_ref[col] = {
            "bin_edges": bin_edges,
            "bin_freq": freq,
            "sample": values[: min(len(values), 1000)],
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    categorical_ref: dict[str, dict[str, Any]] = {}
    for col in MONITORED_CATEGORICAL_COLS:
        if col not in sample_df.columns:
            continue
        counts = sample_df[col].value_counts(normalize=True)
        categorical_ref[col] = {
            "categories": counts.index.tolist(),
            "frequencies": counts.to_numpy(),
        }

    prediction_reference: dict[str, Any] = {}
    if predict_fn is not None and engineer_fn is not None:
        try:
            engineered = engineer_fn(sample_df)
            class_pred, confidence, rul_pred = predict_fn(engineered)
            classes, class_counts = np.unique(class_pred, return_counts=True)
            prediction_reference = {
                "class_labels": classes.tolist(),
                "class_freq": (class_counts / class_counts.sum()).tolist(),
                "mean_confidence": float(np.mean(confidence)),
                "mean_rul": float(np.mean(rul_pred)),
            }
        except Exception:
            logger.exception("Could not build prediction reference (covariate/feature drift still works without it)")

    return DriftReference(
        numeric=numeric_ref,
        categorical=categorical_ref,
        prediction_reference=prediction_reference,
        built_at_utc=pd.Timestamp.utcnow().isoformat(),
        n_reference_rows=len(sample_df),
    )


def save_reference(reference: DriftReference, path: str | Path) -> None:
    joblib.dump(reference, path)
    logger.info("Saved drift reference to %s (%d rows)", path, reference.n_reference_rows)


def load_reference(path: str | Path) -> DriftReference:
    return joblib.load(path)


def _numeric_drift(reference: DriftReference, col: str, live_values: np.ndarray) -> dict:
    ref = reference.numeric[col]
    edges = ref["bin_edges"]
    clipped = np.clip(live_values, edges[0], edges[-1])
    live_counts, _ = np.histogram(clipped, bins=edges)
    live_freq = live_counts / max(live_counts.sum(), 1)
    psi = _psi(ref["bin_freq"], live_freq)
    ks_stat, ks_pvalue = stats.ks_2samp(ref["sample"], live_values)
    return {
        "feature": col,
        "psi": round(psi, 4),
        "drift_status": _psi_band(psi),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "reference_mean": round(ref["mean"], 3),
        "live_mean": round(float(np.mean(live_values)), 3),
        "n_live_samples": int(len(live_values)),
    }


def _categorical_drift(reference: DriftReference, col: str, live_values: pd.Series) -> dict:
    ref = reference.categorical[col]
    ref_categories = ref["categories"]
    live_counts = live_values.value_counts(normalize=True)
    all_categories = list(dict.fromkeys(ref_categories + live_counts.index.tolist()))
    ref_freq = np.array([dict(zip(ref_categories, ref["frequencies"])).get(c, 0.0) for c in all_categories])
    live_freq = np.array([live_counts.get(c, 0.0) for c in all_categories])
    psi = _psi(ref_freq, live_freq)
    return {
        "feature": col,
        "psi": round(psi, 4),
        "drift_status": _psi_band(psi),
        "new_categories": sorted(set(live_counts.index) - set(ref_categories)),
        "n_live_samples": int(len(live_values)),
    }


class DriftMonitor:

    def __init__(self, reference: DriftReference | None, buffer_size: int = DEFAULT_BUFFER_SIZE):
        self.reference = reference
        self._lock = threading.Lock()
        self._inputs: dict[str, deque] = {col: deque(maxlen=buffer_size) for col in MONITORED_NUMERIC_COLS}
        self._categoricals: dict[str, deque] = {col: deque(maxlen=buffer_size) for col in MONITORED_CATEGORICAL_COLS}
        self._class_preds: deque = deque(maxlen=buffer_size)
        self._confidences: deque = deque(maxlen=buffer_size)
        self._ruls: deque = deque(maxlen=buffer_size)

    def record_input(self, row: dict) -> None:
        with self._lock:
            for col in MONITORED_NUMERIC_COLS:
                if col in row and row[col] is not None and not (isinstance(row[col], float) and np.isnan(row[col])):
                    self._inputs[col].append(float(row[col]))
            for col in MONITORED_CATEGORICAL_COLS:
                if col in row and row[col]:
                    self._categoricals[col].append(row[col])

    def record_prediction(self, failure_class: int, confidence: float, rul: float) -> None:
        with self._lock:
            self._class_preds.append(failure_class)
            self._confidences.append(confidence)
            self._ruls.append(rul)

    def compute_drift_report(self, min_samples: int = DEFAULT_MIN_LIVE_SAMPLES) -> dict:
        if self.reference is None:
            return {"status": "unavailable", "reason": "No drift reference loaded for the active model version."}

        with self._lock:
            inputs_snapshot = {col: list(buf) for col, buf in self._inputs.items()}
            categoricals_snapshot = {col: list(buf) for col, buf in self._categoricals.items()}
            class_preds_snapshot = list(self._class_preds)
            confidences_snapshot = list(self._confidences)
            ruls_snapshot = list(self._ruls)

        numeric_reports = []
        for col, values in inputs_snapshot.items():
            if col in self.reference.numeric and len(values) >= min_samples:
                numeric_reports.append(_numeric_drift(self.reference, col, np.array(values)))

        categorical_reports = []
        for col, values in categoricals_snapshot.items():
            if col in self.reference.categorical and len(values) >= min_samples:
                categorical_reports.append(_categorical_drift(self.reference, col, pd.Series(values)))

        prediction_drift = None
        ref_pred = self.reference.prediction_reference
        if ref_pred and len(class_preds_snapshot) >= min_samples:
            live_classes, live_counts = np.unique(class_preds_snapshot, return_counts=True)
            live_freq_map = dict(zip(live_classes.tolist(), (live_counts / live_counts.sum()).tolist()))
            ref_freq_map = dict(zip(ref_pred["class_labels"], ref_pred["class_freq"]))
            all_classes = sorted(set(ref_freq_map) | set(live_freq_map))
            ref_vec = np.array([ref_freq_map.get(c, 0.0) for c in all_classes])
            live_vec = np.array([live_freq_map.get(c, 0.0) for c in all_classes])
            psi = _psi(ref_vec, live_vec)
            prediction_drift = {
                "psi": round(psi, 4),
                "drift_status": _psi_band(psi),
                "reference_mean_confidence": round(ref_pred["mean_confidence"], 4),
                "live_mean_confidence": round(float(np.mean(confidences_snapshot)), 4),
                "reference_mean_rul": round(ref_pred["mean_rul"], 2),
                "live_mean_rul": round(float(np.mean(ruls_snapshot)), 2),
                "n_live_samples": len(class_preds_snapshot),
                "note": "Proxy signal only -- true concept drift requires ground-truth labels this system does not have.",
            }

        all_statuses = [r["drift_status"] for r in numeric_reports + categorical_reports]
        if prediction_drift:
            all_statuses.append(prediction_drift["drift_status"])
        overall = "significant" if "significant" in all_statuses else ("moderate" if "moderate" in all_statuses else "none")

        report = {
            "status": "ok",
            "overall_drift_status": overall,
            "covariate_drift": numeric_reports,
            "categorical_drift": categorical_reports,
            "prediction_drift_proxy": prediction_drift,
            "min_samples_required": min_samples,
        }
        if overall == "significant":
            logger.warning("SIGNIFICANT DATA DRIFT DETECTED: %s", report)
        return report
