"""
The reusable inference pipeline: raw telemetry in, structured predictions
out. FastAPI's `/predict` and `/predict-csv`, the Streamlit dashboard, and
the test suite all call the functions in this module rather than
reimplementing any preprocessing, feature engineering, or model invocation
themselves -- this is the single place "identical to training" is enforced.
"""

from __future__ import annotations

import logging

import pandas as pd

import model_persistence as mp
import pdm_utils as u
from app import feature_engineering, model_loader, monitoring, preprocessing
from app.schemas import PredictionRequest, PredictionResponse
from app.utils import InferenceError, get_prediction_logger, timed

logger = logging.getLogger(__name__)
pred_logger = get_prediction_logger()


def load_artifacts(force_reload: bool = False) -> mp.ModelArtifacts:
    """Load (or return the cached) model artifacts.

    A thin re-export of `app.model_loader.get_artifacts` under the name
    Phase 2 calls for, so this module is a complete, self-contained
    pipeline a caller can import without also knowing about model_loader.
    """
    return model_loader.get_artifacts(force_reload=force_reload)


def preprocess_input(df: pd.DataFrame, artifacts: mp.ModelArtifacts) -> tuple[pd.DataFrame, list[str]]:
    """Sort chronologically per asset, impute missing telemetry, and encode
    Machine_Model. Returns the cleaned DataFrame plus non-fatal warnings.
    """
    df = u.parse_and_sort_timestamps(df)  # rolling/lag features require true chronological order
    df = preprocessing.impute_missing(df)
    df, warnings = preprocessing.safe_encode_machine_model(df, artifacts.encoders)
    return df, warnings


def engineer_features(df: pd.DataFrame, artifacts: mp.ModelArtifacts) -> pd.DataFrame:
    """Build the engineered feature matrix used by both models."""
    return feature_engineering.engineer(df, artifacts.feature_engineering_config, artifacts.feature_list)


def predict_failure(X: pd.DataFrame, artifacts: mp.ModelArtifacts) -> pd.DataFrame:
    """Classify failure severity for every row in `X`.

    Returns one row per input row with the predicted class, label,
    confidence (max class probability), and the full probability
    distribution -- callers pick what they need.
    """
    X_ordered = X[artifacts.feature_list]
    try:
        class_pred = artifacts.classifier.predict(X_ordered)
        proba = artifacts.classifier.predict_proba(X_ordered)
    except Exception as exc:
        raise InferenceError(f"Classifier failed: {exc}") from exc

    confidence = proba.max(axis=1)
    labels = [artifacts.class_names[c] for c in class_pred]
    proba_dicts = [
        {artifacts.class_names[i]: float(p) for i, p in enumerate(row)} for row in proba
    ]
    return pd.DataFrame(
        {
            "Failure_Class_Predicted": class_pred,
            "Failure_State_Predicted": labels,
            "Confidence": confidence,
            "Class_Probabilities": proba_dicts,
        },
        index=X.index,
    )


def predict_rul(X: pd.DataFrame, artifacts: mp.ModelArtifacts) -> pd.Series:
    """Regress remaining useful life (hours) for every row in `X`."""
    X_ordered = X[artifacts.feature_list]
    try:
        rul_pred = artifacts.regressor.predict(X_ordered)
    except Exception as exc:
        raise InferenceError(f"Regressor failed: {exc}") from exc
    return pd.Series(rul_pred, index=X.index, name="RUL_Hours_Predicted").clip(lower=0)


def predict(request: PredictionRequest) -> PredictionResponse:
    """Single-prediction entry point: a reading history in, one
    `PredictionResponse` out for the most recent reading in that history.
    """
    artifacts = load_artifacts()
    with timed(logger, "single prediction"):
        raw_df = preprocessing.request_to_dataframe(request)
        clean_df, warnings = preprocess_input(raw_df, artifacts)
        engineered_df = engineer_features(clean_df, artifacts)
        latest = feature_engineering.latest_reading_per_asset(engineered_df)

        class_result = predict_failure(latest, artifacts)
        rul_result = predict_rul(latest, artifacts)
        row = class_result.iloc[0]

        response = PredictionResponse(
            asset_id=str(latest["Asset_ID"].iloc[0]),
            timestamp=str(latest["Timestamp"].iloc[0]),
            failure_state=row["Failure_State_Predicted"],
            failure_class=int(row["Failure_Class_Predicted"]),
            confidence=round(float(row["Confidence"]), 4),
            remaining_useful_life=round(float(rul_result.iloc[0]), 2),
            class_probabilities={k: round(v, 4) for k, v in row["Class_Probabilities"].items()},
            warnings=warnings,
        )

    pred_logger.info(
        "asset=%s failure_state=%s confidence=%.4f rul=%.2f",
        response.asset_id, response.failure_state, response.confidence, response.remaining_useful_life,
    )
    monitoring.record_prediction()
    return response


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Batch entry point: raw telemetry DataFrame in, one prediction row per
    INPUT row out (used by `/predict-csv` -- every row gets scored, not just
    the latest reading per asset).
    """
    artifacts = load_artifacts()
    with timed(logger, f"batch prediction ({len(df)} rows)"):
        clean_df, warnings = preprocess_input(df, artifacts)
        engineered_df = engineer_features(clean_df, artifacts)

        class_result = predict_failure(engineered_df, artifacts)
        rul_result = predict_rul(engineered_df, artifacts)

        output = pd.DataFrame(
            {
                "Asset_ID": engineered_df["Asset_ID"],
                "Timestamp": engineered_df["Timestamp"],
                "Failure_State_Predicted": class_result["Failure_State_Predicted"],
                "Confidence": class_result["Confidence"].round(4),
                "RUL_Hours_Predicted": rul_result.round(2),
            }
        ).reset_index(drop=True)

    for w in warnings:
        logger.warning(w)
    pred_logger.info("batch prediction: %d rows scored", len(output))
    for _ in range(len(output)):
        monitoring.record_prediction()
    return output
