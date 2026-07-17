from __future__ import annotations

import logging
import time

import pandas as pd

import model_persistence as mp
import pdm_utils as u
from app import fault_diagnosis, feature_engineering, model_loader, monitoring, preprocessing
from app.audit_log import AuditRecord
from app.schemas import FaultDiagnosis, PredictionRequest, PredictionResponse
from app.utils import InferenceError, get_prediction_logger, timed

logger = logging.getLogger(__name__)
pred_logger = get_prediction_logger()


def load_artifacts(force_reload: bool = False) -> mp.ModelArtifacts:
    return model_loader.get_artifacts(force_reload=force_reload)


def preprocess_input(df: pd.DataFrame, artifacts: mp.ModelArtifacts) -> tuple[pd.DataFrame, list[str]]:
    df = u.parse_and_sort_timestamps(df)
    df = preprocessing.impute_missing(df)
    df, warnings = preprocessing.safe_encode_categoricals(df, artifacts.encoders, artifacts.feature_engineering_config)
    return df, warnings


def engineer_features(df: pd.DataFrame, artifacts: mp.ModelArtifacts) -> pd.DataFrame:
    return feature_engineering.engineer(df, artifacts.feature_engineering_config, artifacts.feature_list)


def predict_failure(X: pd.DataFrame, artifacts: mp.ModelArtifacts) -> pd.DataFrame:
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
    X_ordered = X[artifacts.feature_list]
    try:
        rul_pred = artifacts.regressor.predict(X_ordered)
    except Exception as exc:
        raise InferenceError(f"Regressor failed: {exc}") from exc
    return pd.Series(rul_pred, index=X.index, name="RUL_Hours_Predicted").clip(lower=0)


CLASS_RUL_BOUNDS = {
    "Normal": (51, 500),
    "Warning": (11, 50),
    "Critical": (0, 10),
}


def apply_rul_consistency_rule(
    failure_states: pd.Series, rul: pd.Series, diagnosed_faults: pd.Series | None = None,
) -> pd.Series:
    clamped = rul.copy()
    for state, (lo, hi) in CLASS_RUL_BOUNDS.items():
        mask = failure_states == state
        if mask.any():
            clamped.loc[mask] = rul.loc[mask].clip(lo, hi)

    if diagnosed_faults is not None:
        already_failed_mask = (failure_states == "Critical") & diagnosed_faults.isin(fault_diagnosis.SPECIFIC_FAULTS)
        clamped = clamped.where(~already_failed_mask, 0.0)

    return clamped


def diagnose_at_risk_rows(
    df: pd.DataFrame, failure_states: pd.Series, baseline: dict,
) -> tuple[pd.Series, dict[int, fault_diagnosis.FaultDiagnosisResult]]:
    diagnosed_faults = pd.Series([None] * len(df), index=df.index, dtype=object)
    diagnoses: dict[int, fault_diagnosis.FaultDiagnosisResult] = {}

    at_risk_mask = failure_states.isin(("Warning", "Critical"))
    for idx in df.index[at_risk_mask.to_numpy()]:
        result = fault_diagnosis.diagnose(df.loc[idx], baseline)
        diagnosed_faults.loc[idx] = result.predicted_fault
        diagnoses[idx] = result

    return diagnosed_faults, diagnoses


def _predict_core(request: PredictionRequest, artifacts: mp.ModelArtifacts) -> tuple[PredictionResponse, pd.DataFrame]:
    start = time.perf_counter()
    raw_df = preprocessing.request_to_dataframe(request)
    clean_df, warnings = preprocess_input(raw_df, artifacts)
    engineered_df = engineer_features(clean_df, artifacts)
    latest = feature_engineering.latest_reading_per_asset(engineered_df)

    class_result = predict_failure(latest, artifacts)
    diagnosed_faults, diagnoses = diagnose_at_risk_rows(
        latest, class_result["Failure_State_Predicted"], artifacts.fault_diagnosis_baseline
    )
    rul_result = predict_rul(latest, artifacts)
    rul_result = apply_rul_consistency_rule(class_result["Failure_State_Predicted"], rul_result, diagnosed_faults)
    row = class_result.iloc[0]
    row_index = latest.index[0]
    latency_ms = (time.perf_counter() - start) * 1000

    diagnosis_result = diagnoses.get(row_index)
    response = PredictionResponse(
        asset_id=str(latest["Asset_ID"].iloc[0]),
        timestamp=str(latest["Timestamp"].iloc[0]),
        failure_state=row["Failure_State_Predicted"],
        failure_class=int(row["Failure_Class_Predicted"]),
        confidence=round(float(row["Confidence"]), 4),
        remaining_useful_life=round(float(rul_result.iloc[0]), 2),
        class_probabilities={k: round(v, 4) for k, v in row["Class_Probabilities"].items()},
        warnings=warnings,
        fault_diagnosis=FaultDiagnosis(**vars(diagnosis_result)) if diagnosis_result else None,
    )

    pred_logger.info(
        "asset=%s failure_state=%s confidence=%.4f rul=%.2f",
        response.asset_id, response.failure_state, response.confidence, response.remaining_useful_life,
    )
    monitoring.record_prediction(confidence=response.confidence)
    drift_monitor = model_loader.get_drift_monitor()
    drift_monitor.record_input(latest.iloc[0].to_dict())
    drift_monitor.record_prediction(response.failure_class, response.confidence, response.remaining_useful_life)

    model_loader.get_audit_logger().log_prediction(AuditRecord(
        timestamp=response.timestamp,
        asset_id=response.asset_id,
        input_features={col: latest[col].iloc[0] for col in u.RAW_NUMERIC_COLS + ["Machine_Model", "Location"] if col in latest.columns},
        predicted_class=response.failure_class,
        predicted_state=response.failure_state,
        confidence=response.confidence,
        rul_hours=response.remaining_useful_life,
        model_version=model_loader.get_active_version() or "unknown",
        inference_latency_ms=round(latency_ms, 2),
    ))
    return response, latest


def predict(request: PredictionRequest) -> PredictionResponse:
    artifacts = load_artifacts()
    with timed(logger, "single prediction"):
        response, _ = _predict_core(request, artifacts)
    return response


def explain(request: PredictionRequest) -> dict:
    from app import explainability as xai

    artifacts = load_artifacts()
    with timed(logger, "prediction + explanation"):
        response, latest = _predict_core(request, artifacts)
        explainers = model_loader.get_explainers()
        explanation = xai.explain_prediction(
            latest, explainers, artifacts.feature_list, artifacts.class_names,
            predicted_class=response.failure_class,
        )

    return {
        "prediction": response,
        "predicted_class_label": explanation["predicted_class_label"],
        "classifier_top_features": explanation["classifier_top_features"],
        "regressor_top_features": explanation["regressor_top_features"],
        "classifier_base_value": explanation["classifier_base_value"],
        "regressor_base_value": explanation["regressor_base_value"],
        "classifier_waterfall_png_base64": explanation["classifier_waterfall_png_base64"],
        "regressor_waterfall_png_base64": explanation["regressor_waterfall_png_base64"],
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = load_artifacts()
    start = time.perf_counter()
    with timed(logger, f"batch prediction ({len(df)} rows)"):
        clean_df, warnings = preprocess_input(df, artifacts)
        engineered_df = engineer_features(clean_df, artifacts)

        class_result = predict_failure(engineered_df, artifacts)
        diagnosed_faults, _ = diagnose_at_risk_rows(
            engineered_df, class_result["Failure_State_Predicted"], artifacts.fault_diagnosis_baseline
        )
        rul_result = predict_rul(engineered_df, artifacts)
        rul_result = apply_rul_consistency_rule(class_result["Failure_State_Predicted"], rul_result, diagnosed_faults)

        output_columns = {
            "Asset_ID": engineered_df["Asset_ID"],
            "Timestamp": engineered_df["Timestamp"],
        }
        for col in ["Machine_Model", "Location"] + u.RAW_NUMERIC_COLS:
            if col in engineered_df.columns:
                output_columns[col] = engineered_df[col]
        output_columns.update({
            "Failure_State_Predicted": class_result["Failure_State_Predicted"],
            "Confidence": class_result["Confidence"].round(4),
            "RUL_Hours_Predicted": rul_result.round(2),
            "Diagnosed_Fault": diagnosed_faults.fillna(""),
        })
        output = pd.DataFrame(output_columns).reset_index(drop=True)
    per_row_latency_ms = (time.perf_counter() - start) * 1000 / max(len(output), 1)

    for w in warnings:
        logger.warning(w)
    pred_logger.info("batch prediction: %d rows scored", len(output))

    drift_monitor = model_loader.get_drift_monitor()
    audit_logger = model_loader.get_audit_logger()
    active_version = model_loader.get_active_version() or "unknown"
    input_cols = [c for c in u.RAW_NUMERIC_COLS + ["Machine_Model", "Location"] if c in engineered_df.columns]

    for (_, eng_row), (_, class_row), rul in zip(
        engineered_df.iterrows(), class_result.iterrows(), rul_result
    ):
        drift_monitor.record_input(eng_row.to_dict())
        monitoring.record_prediction(confidence=float(class_row["Confidence"]))
        drift_monitor.record_prediction(int(class_row["Failure_Class_Predicted"]), float(class_row["Confidence"]), float(rul))
        audit_logger.log_prediction(AuditRecord(
            timestamp=str(eng_row["Timestamp"]),
            asset_id=str(eng_row["Asset_ID"]),
            input_features={col: eng_row[col] for col in input_cols},
            predicted_class=int(class_row["Failure_Class_Predicted"]),
            predicted_state=class_row["Failure_State_Predicted"],
            confidence=float(class_row["Confidence"]),
            rul_hours=float(rul),
            model_version=active_version,
            inference_latency_ms=round(per_row_latency_ms, 2),
        ))

    return output
