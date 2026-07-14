"""
Reproduces Stage 4's engineered features (rolling stats, lags, rate/percent
change, interactions) at inference time, driven by the persisted
`feature_engineering_config` rather than re-hardcoding window sizes -- so if
Stage 4 ever changes its windows and the artifacts are re-saved, this module
picks up the new config automatically without a code change.
"""

from __future__ import annotations

import logging

import pandas as pd

import pdm_utils as u
from app.utils import InputValidationError

logger = logging.getLogger(__name__)


def engineer(df: pd.DataFrame, feature_engineering_config: dict, feature_list: list[str]) -> pd.DataFrame:
    """Apply the full engineered-feature pipeline and select the trained
    feature columns, in the exact order the models expect.

    `df` must already have Timestamp parsed/sorted, missing values imputed,
    and `Machine_Model_Encoded` present (see `app/preprocessing.py`).
    """
    cfg = feature_engineering_config
    raw_cols = cfg["raw_numeric_cols"]

    df = u.add_rolling_features(df, raw_cols, windows=cfg["rolling_windows"])
    df = u.add_lag_features(df, raw_cols, lags=cfg["lag_steps"])
    df = u.add_rate_and_pct_change(df, raw_cols)
    df = u.add_interaction_features(df)  # fixed 4 pairs, matches training exactly

    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        # Should be unreachable if feature_engineering_config matches the
        # feature_list that was saved alongside it -- surfaced loudly rather
        # than silently dropping columns the model expects.
        raise InputValidationError(f"Feature engineering did not produce required column(s): {missing}")

    return df


def latest_reading_per_asset(engineered_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent (last, post-sort) row per Asset_ID.

    Used for the single-prediction endpoint: a submitted history is only
    there to give rolling/lag features real context -- the actual answer
    the caller wants is "what does the pump look like right now."
    """
    return engineered_df.sort_values("Timestamp").groupby("Asset_ID", as_index=False).tail(1)
