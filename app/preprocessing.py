from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import pdm_utils as u
from app.schemas import PredictionRequest
from app.utils import InputValidationError

logger = logging.getLogger(__name__)

_DEFAULT_ASSET_ID = "UNKNOWN_ASSET"

_FIELD_TO_COLUMN = {
    "timestamp": "Timestamp",
    "asset_id": "Asset_ID",
    "machine_model": "Machine_Model",
    "location": "Location",
    "vibration_mm_s": "Vibration_mm_s",
    "temperature_c": "Temperature_C",
    "pressure_psi": "Pressure_psi",
    "flow_rate_m3_h": "Flow_Rate_m3_h",
}


def request_to_dataframe(request: PredictionRequest) -> pd.DataFrame:
    records = [r.model_dump() for r in request.readings]
    df = pd.DataFrame(records).rename(columns=_FIELD_TO_COLUMN)

    if df["Asset_ID"].isnull().any():
        df["Asset_ID"] = df["Asset_ID"].fillna(_DEFAULT_ASSET_ID)

    if df["Timestamp"].isnull().any():
        logger.info("Some readings had no timestamp; assigning sequential order.")
        synthetic = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="h")
        df.loc[df["Timestamp"].isnull(), "Timestamp"] = synthetic[df["Timestamp"].isnull()]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    return df


def dataframe_from_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    except Exception as exc:
        raise InputValidationError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise InputValidationError("Uploaded CSV has no rows.")

    df = df.rename(columns={k: v for k, v in _FIELD_TO_COLUMN.items() if k in df.columns})

    missing_required = {"Machine_Model", "Location"} - set(df.columns)
    if missing_required:
        raise InputValidationError(f"CSV is missing required column(s): {sorted(missing_required)}")

    telemetry_cols = [c for c in u.RAW_NUMERIC_COLS if c in df.columns]
    if not telemetry_cols:
        raise InputValidationError(
            f"CSV has none of the expected telemetry columns: {u.RAW_NUMERIC_COLS}"
        )

    if "Asset_ID" not in df.columns:
        df["Asset_ID"] = [f"{_DEFAULT_ASSET_ID}_{i}" for i in range(len(df))]
    elif df["Asset_ID"].isnull().any():
        raise InputValidationError(
            "CSV contains missing Asset_ID values -- either provide an Asset_ID for every row "
            "or omit the column entirely so one is generated for you."
        )
    if df["Location"].isnull().any():
        raise InputValidationError("CSV contains missing Location values.")
    if "Timestamp" not in df.columns:
        df["Timestamp"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="h")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].isnull().any():
        raise InputValidationError("CSV contains unparseable Timestamp values.")

    for col in u.RAW_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_of_range_errors = []
    for col, (lo, hi) in u.VALID_RANGES.items():
        values = df[col]
        bad_mask = values.notnull() & ((values < lo) | (values > hi))
        if bad_mask.any():
            bad_values = sorted(values[bad_mask].unique().tolist())
            out_of_range_errors.append(f"{col} must be within [{lo}, {hi}], got {bad_values}")
    if out_of_range_errors:
        raise InputValidationError(
            "CSV contains sensor readings outside the physically valid range "
            f"the models were trained on: {'; '.join(out_of_range_errors)}"
        )

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = u.handle_missing_values(df, u.RAW_NUMERIC_COLS)
    still_missing = df[u.RAW_NUMERIC_COLS].isnull().any()
    if still_missing.any():
        bad_cols = still_missing[still_missing].index.tolist()
        raise InputValidationError(
            f"Could not impute missing values for column(s) {bad_cols}: "
            "no history and no valid values anywhere in the request to interpolate from."
        )
    return df


def safe_encode_categoricals(
    df: pd.DataFrame, encoders: u.LabelEncoders, feature_engineering_config: dict,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    warnings: list[str] = []
    categorical_cols = feature_engineering_config.get("categorical_cols", ["Machine_Model"])

    for col in categorical_cols:
        if col not in df.columns or col not in encoders.encoders:
            continue
        le = encoders.encoders[col]
        known = set(le.classes_)
        fallback_label = str(le.classes_[0])

        str_col = df[col].astype(str)
        unknown_mask = ~str_col.isin(known)
        if unknown_mask.any():
            unknown_values = sorted(str_col[unknown_mask].unique())
            msg = f"Unknown {col} value(s) {unknown_values} -- substituted '{fallback_label}' for encoding."
            logger.warning(msg)
            warnings.append(msg)
            str_col = str_col.where(~unknown_mask, fallback_label)

        df[f"{col}_Encoded"] = le.transform(str_col)

    return df, warnings
