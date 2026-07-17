from __future__ import annotations

import json
import logging
import os
import tempfile
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
    logger.propagate = False

DEFAULT_MODELS_DIR = Path("models")

CLASSIFIER_FILENAME_TEMPLATE = "classification_model_{name}.joblib"
REGRESSOR_FILENAME_TEMPLATE = "regression_model_{name}.joblib"
SCALER_FILENAME = "feature_scaler.joblib"
ENCODERS_FILENAME = "label_encoders.joblib"
FEATURE_LIST_FILENAME = "selected_features.joblib"
CLASS_NAMES_FILENAME = "class_names.joblib"
FEATURE_ENGINEERING_CONFIG_FILENAME = "feature_engineering_config.joblib"
FAULT_DIAGNOSIS_BASELINE_FILENAME = "fault_diagnosis_baseline.json"
METADATA_FILENAME = "metadata.json"


class ModelPersistenceError(RuntimeError):
    pass


def atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


@dataclass
class ModelArtifacts:

    classifier: Any
    classifier_name: str
    regressor: Any
    regressor_name: str
    scaler: Any
    encoders: Any
    feature_list: list[str]
    class_names: list[str]
    feature_engineering_config: dict
    fault_diagnosis_baseline: dict
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
    fault_diagnosis_baseline: dict | None = None,
) -> dict[str, Path]:
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

        baseline_path = models_dir / FAULT_DIAGNOSIS_BASELINE_FILENAME
        atomic_write_text(baseline_path, json.dumps(fault_diagnosis_baseline or {}, indent=2))
        saved_paths[FAULT_DIAGNOSIS_BASELINE_FILENAME] = baseline_path
        logger.info("Saved %s", FAULT_DIAGNOSIS_BASELINE_FILENAME)

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
        atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
        saved_paths[METADATA_FILENAME] = metadata_path
        logger.info("Saved %s", METADATA_FILENAME)

    except Exception as exc:
        logger.exception("Failed to save model artifacts")
        raise ModelPersistenceError(f"save_models failed: {exc}") from exc

    logger.info("All %d artifacts saved successfully.", len(saved_paths))
    return saved_paths


def load_models(models_dir: str | Path = DEFAULT_MODELS_DIR) -> ModelArtifacts:
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

        baseline_path = models_dir / FAULT_DIAGNOSIS_BASELINE_FILENAME
        fault_diagnosis_baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

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
        fault_diagnosis_baseline=fault_diagnosis_baseline,
        metadata=metadata,
    )


def predict(artifacts: ModelArtifacts, X: pd.DataFrame) -> pd.DataFrame:
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
