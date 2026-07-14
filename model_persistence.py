"""
Model persistence layer for the Predictive Maintenance (PdM) project.

Serializes every artifact required to reproduce inference-time predictions
without retraining: the two selected models plus the fitted preprocessing
objects and metadata needed to rebuild a feature vector in the exact shape
the models were trained on.

This module trains nothing. It only saves objects handed to it by the
training notebook (already-fitted estimators) and reloads them later. It has
no dependency on the notebook, pandas globals, or any other project module,
so it can be imported unchanged by a future FastAPI service or Streamlit app.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    # Standalone-usage handler (e.g. run from the notebook directly). When
    # this module is imported into a larger app that configures its own
    # root logging (see app.utils.setup_logging), disable propagation so
    # records aren't printed twice -- once here, once by the root handler.
    logger.propagate = False

DEFAULT_MODELS_DIR = Path("models")

# Fixed, meaningful filenames shared by save_models() and load_models().
CLASSIFIER_FILENAME_TEMPLATE = "classification_model_{name}.joblib"
REGRESSOR_FILENAME_TEMPLATE = "regression_model_{name}.joblib"
SCALER_FILENAME = "feature_scaler.joblib"
ENCODERS_FILENAME = "label_encoders.joblib"
FEATURE_LIST_FILENAME = "selected_features.joblib"
CLASS_NAMES_FILENAME = "class_names.joblib"
FEATURE_ENGINEERING_CONFIG_FILENAME = "feature_engineering_config.joblib"
METADATA_FILENAME = "metadata.json"


class ModelPersistenceError(RuntimeError):
    """Raised when saving or loading a model artifact set fails."""


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


@dataclass
class ModelArtifacts:
    """Everything a serving layer (FastAPI / Streamlit) needs for inference.

    Returned by `load_models()` as a single object so callers don't have to
    juggle six separate file paths.
    """

    classifier: Any
    classifier_name: str
    regressor: Any
    regressor_name: str
    scaler: Any
    encoders: Any
    feature_list: list[str]
    class_names: list[str]
    feature_engineering_config: dict
    metadata: dict


def save_models(
    classifier: Any,
    classifier_name: str,
    regressor: Any,
    regressor_name: str,
    scaler: Any,
    encoders: Any,
    feature_list: list[str],
    class_names: list[str],
    feature_engineering_config: dict,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    extra_metadata: dict | None = None,
) -> dict[str, Path]:
    """Persist every inference artifact via joblib.

    Callers pass in already-fitted objects from the training run -- this
    function does not fit or train anything. Raises `ModelPersistenceError`
    on any failure rather than leaving a partially-written artifact set on
    disk silently believed to be complete.

    Returns a dict mapping each saved filename to its full path.
    """
    models_dir = Path(models_dir)
    saved_paths: dict[str, Path] = {}
    classifier_filename = CLASSIFIER_FILENAME_TEMPLATE.format(name=_slug(classifier_name))
    regressor_filename = REGRESSOR_FILENAME_TEMPLATE.format(name=_slug(regressor_name))

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving model artifacts to %s", models_dir.resolve())

        artifacts = {
            classifier_filename: classifier,
            regressor_filename: regressor,
            SCALER_FILENAME: scaler,
            ENCODERS_FILENAME: encoders,
            FEATURE_LIST_FILENAME: list(feature_list),
            CLASS_NAMES_FILENAME: list(class_names),
            FEATURE_ENGINEERING_CONFIG_FILENAME: dict(feature_engineering_config),
        }
        for filename, obj in artifacts.items():
            path = models_dir / filename
            joblib.dump(obj, path)
            saved_paths[filename] = path
            logger.info("Saved %s (%s)", filename, type(obj).__name__)

        metadata = {
            "classifier_name": classifier_name,
            "classifier_filename": classifier_filename,
            "regressor_name": regressor_name,
            "regressor_filename": regressor_filename,
            "n_features": len(feature_list),
            "class_names": list(class_names),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "joblib_version": joblib.__version__,
            **(extra_metadata or {}),
        }
        metadata_path = models_dir / METADATA_FILENAME
        metadata_path.write_text(json.dumps(metadata, indent=2))
        saved_paths[METADATA_FILENAME] = metadata_path
        logger.info("Saved %s", METADATA_FILENAME)

    except Exception as exc:
        logger.exception("Failed to save model artifacts")
        raise ModelPersistenceError(f"save_models failed: {exc}") from exc

    logger.info("All %d artifacts saved successfully.", len(saved_paths))
    return saved_paths


def load_models(models_dir: str | Path = DEFAULT_MODELS_DIR) -> ModelArtifacts:
    """Load everything `save_models()` wrote and return it as one object.

    Raises `ModelPersistenceError` if the directory, metadata file, or any
    artifact the metadata references is missing or corrupt -- a serving
    layer should fail loudly at startup rather than run with a partial,
    silently-broken artifact set.
    """
    models_dir = Path(models_dir)

    def _load(filename: str):
        path = models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected artifact missing: {path}")
        obj = joblib.load(path)
        logger.info("Loaded %s (%s)", filename, type(obj).__name__)
        return obj

    try:
        metadata_path = models_dir / METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"{metadata_path} not found -- run save_models() first")
        metadata = json.loads(metadata_path.read_text())
        logger.info(
            "Loading model artifacts from %s (saved %s)",
            models_dir.resolve(), metadata.get("saved_at_utc"),
        )

        classifier = _load(metadata["classifier_filename"])
        regressor = _load(metadata["regressor_filename"])
        scaler = _load(SCALER_FILENAME)
        encoders = _load(ENCODERS_FILENAME)
        feature_list = _load(FEATURE_LIST_FILENAME)
        class_names = _load(CLASS_NAMES_FILENAME)
        feature_engineering_config = _load(FEATURE_ENGINEERING_CONFIG_FILENAME)

    except Exception as exc:
        logger.exception("Failed to load model artifacts")
        raise ModelPersistenceError(f"load_models failed: {exc}") from exc

    logger.info("All artifacts loaded successfully.")
    return ModelArtifacts(
        classifier=classifier,
        classifier_name=metadata["classifier_name"],
        regressor=regressor,
        regressor_name=metadata["regressor_name"],
        scaler=scaler,
        encoders=encoders,
        feature_list=feature_list,
        class_names=class_names,
        feature_engineering_config=feature_engineering_config,
        metadata=metadata,
    )


def predict(artifacts: ModelArtifacts, X: pd.DataFrame) -> pd.DataFrame:
    """Run both models on a raw feature frame and return labeled predictions.

    `X` must already contain engineered features in their raw (unscaled)
    units -- the classifier and regressor were both trained directly on
    unscaled features (they're tree-based and scale-invariant), so
    `artifacts.scaler` is intentionally NOT applied here. Columns are
    reordered to `artifacts.feature_list` so callers don't need to match
    training's exact column order themselves.

    This is the single function a FastAPI endpoint or Streamlit callback
    should call after building a feature row from raw telemetry.
    """
    try:
        X_ordered = X[artifacts.feature_list]
    except KeyError as exc:
        raise ModelPersistenceError(
            f"Input is missing required feature columns: {exc}"
        ) from exc

    class_pred = artifacts.classifier.predict(X_ordered)
    rul_pred = artifacts.regressor.predict(X_ordered)

    return pd.DataFrame({
        "Failure_Class_Predicted": class_pred,
        "Failure_State_Predicted": [artifacts.class_names[c] for c in class_pred],
        "RUL_Hours_Predicted": rul_pred,
    }, index=X.index)
