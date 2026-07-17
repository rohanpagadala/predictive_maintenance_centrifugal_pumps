from __future__ import annotations

import logging

import pandas as pd

import pdm_utils as u
from app.utils import InputValidationError

logger = logging.getLogger(__name__)


def engineer(df: pd.DataFrame, feature_engineering_config: dict, feature_list: list[str]) -> pd.DataFrame:
    cfg = feature_engineering_config
    try:
        raw_cols = cfg["raw_numeric_cols"]
        rolling_windows = cfg["rolling_windows"]
        lag_steps = cfg["lag_steps"]
    except KeyError as exc:
        raise InputValidationError(
            f"Model artifacts have a malformed feature_engineering_config, missing key: {exc}"
        ) from exc

    df = u.add_rolling_features(df, raw_cols, windows=rolling_windows)
    df = u.add_lag_features(df, raw_cols, lags=lag_steps)
    df = u.add_rate_and_pct_change(df, raw_cols)
    df = u.add_interaction_features(df)

    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise InputValidationError(f"Feature engineering did not produce required column(s): {missing}")

    return df


def latest_reading_per_asset(engineered_df: pd.DataFrame) -> pd.DataFrame:
    return engineered_df.sort_values("Timestamp").groupby("Asset_ID", as_index=False).tail(1)
