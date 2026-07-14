"""
Raw-input preprocessing for inference: turns validated Pydantic readings into
the same shape `pdm_utils` expects (Timestamp parsed & sorted, missing values
imputed, Machine_Model encoded) -- reusing `pdm_utils` functions directly so
"identical to training" is true by construction, not by re-implementation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import pdm_utils as u
from app.schemas import PredictionRequest
from app.utils import InputValidationError

logger = logging.getLogger(__name__)

_DEFAULT_ASSET_ID = "UNKNOWN_ASSET"

# Maps the API's snake_case field names to the PascalCase columns pdm_utils
# and the trained models expect.
_FIELD_TO_COLUMN = {
    "timestamp": "Timestamp",
    "asset_id": "Asset_ID",
    "machine_model": "Machine_Model",
    "vibration_mm_s": "Vibration_mm_s",
    "temperature_c": "Temperature_C",
    "pressure_psi": "Pressure_psi",
    "flow_rate_m3_h": "Flow_Rate_m3_h",
}


def request_to_dataframe(request: PredictionRequest) -> pd.DataFrame:
    """Convert a validated `PredictionRequest` into a pdm_utils-shaped DataFrame.

    Missing `asset_id` defaults to a shared placeholder (single-pump request);
    missing `timestamp` values are back-filled with a synthetic, strictly
    increasing sequence so submission order is preserved as chronological
    order -- pdm_utils' rolling/lag features depend on that ordering.
    """
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
    """Parse an uploaded CSV into the same pdm_utils-shaped DataFrame.

    Accepts either the API's snake_case headers or the training data's
    PascalCase headers, so a user can re-upload an export of the original
    dataset without renaming columns by hand.
    """
    try:
        df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    except Exception as exc:
        raise InputValidationError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise InputValidationError("Uploaded CSV has no rows.")

    df = df.rename(columns={k: v for k, v in _FIELD_TO_COLUMN.items() if k in df.columns})

    missing_required = {"Machine_Model"} - set(df.columns)
    if missing_required:
        raise InputValidationError(f"CSV is missing required column(s): {sorted(missing_required)}")

    telemetry_cols = [c for c in u.RAW_NUMERIC_COLS if c in df.columns]
    if not telemetry_cols:
        raise InputValidationError(
            f"CSV has none of the expected telemetry columns: {u.RAW_NUMERIC_COLS}"
        )

    if "Asset_ID" not in df.columns:
        # Each CSV row is treated as an independent pump reading with no
        # history -- assign a unique synthetic Asset_ID per row so rows
        # aren't accidentally grouped together as one pump's history.
        df["Asset_ID"] = [f"{_DEFAULT_ASSET_ID}_{i}" for i in range(len(df))]
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

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing telemetry via pdm_utils' per-asset interpolation.

    Raises `InputValidationError` if any telemetry column remains entirely
    null for an asset after imputation (nothing to interpolate from and no
    batch-level median to fall back on) -- silently feeding NaN into a tree
    model produces a nonsensical prediction, so this fails loudly instead.
    """
    df = u.handle_missing_values(df, u.RAW_NUMERIC_COLS)
    still_missing = df[u.RAW_NUMERIC_COLS].isnull().any()
    if still_missing.any():
        bad_cols = still_missing[still_missing].index.tolist()
        raise InputValidationError(
            f"Could not impute missing values for column(s) {bad_cols}: "
            "no history and no valid values anywhere in the request to interpolate from."
        )
    return df


def safe_encode_machine_model(df: pd.DataFrame, encoders: u.LabelEncoders) -> tuple[pd.DataFrame, list[str]]:
    """Encode Machine_Model with the fitted training encoder, substituting a
    fallback code (and a warning) for any model name never seen in training.

    `LabelEncoder.transform` raises on unseen labels; re-fitting a new
    encoder at inference time would silently assign different integers than
    training used, corrupting the feature -- so unseen values are mapped to
    the most frequent training class instead, which is the safer default.
    """
    df = df.copy()
    warnings: list[str] = []
    le = encoders.encoders["Machine_Model"]
    known = set(le.classes_)
    fallback_label = str(le.classes_[0])  # classes_ is sorted; stable, deterministic fallback

    unknown_mask = ~df["Machine_Model"].astype(str).isin(known)
    if unknown_mask.any():
        unknown_values = sorted(df.loc[unknown_mask, "Machine_Model"].astype(str).unique())
        msg = f"Unknown machine_model value(s) {unknown_values} -- substituted '{fallback_label}' for encoding."
        logger.warning(msg)
        warnings.append(msg)
        df.loc[unknown_mask, "Machine_Model"] = fallback_label

    df["Machine_Model_Encoded"] = le.transform(df["Machine_Model"].astype(str))
    return df, warnings
