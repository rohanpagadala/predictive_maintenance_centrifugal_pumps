"""
Reusable pipeline functions for the Predictive Maintenance (PdM) project.

Organized into stages that mirror the project workflow:
  - Data loading
  - EDA helpers
  - Preprocessing
  - Feature engineering
  - Feature selection
  - Data preparation (train/test split)
  - Model training & evaluation

All functions are pure (no hidden global state) so they can be unit-tested
and re-used from the notebook, a future training script, or an API layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

RANDOM_STATE = 42

RAW_NUMERIC_COLS = [
    "Vibration_mm_s",
    "Temperature_C",
    "Pressure_psi",
    "Flow_Rate_m3_h",
]

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw pump telemetry file (CSV or Excel) into a DataFrame."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------


def parse_and_sort_timestamps(
    df: pd.DataFrame, asset_col: str = "Asset_ID", time_col: str = "Timestamp"
) -> pd.DataFrame:
    """Convert Timestamp to datetime and sort chronologically within each asset.

    Sorting is done by (asset, time) rather than time alone because rolling /
    lag features are computed per-asset later on: each physical pump must see
    its own readings in chronological order, independent of how other pumps'
    readings are interleaved in the raw file.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([asset_col, time_col]).reset_index(drop=True)
    return df


def drop_duplicates_and_report(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows, returning the cleaned frame and drop count."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, n_before - len(df)


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    asset_col: str = "Asset_ID",
    time_col: str = "Timestamp",
) -> pd.DataFrame:
    """Impute missing telemetry via per-asset time interpolation.

    Straight mean/median imputation would ignore that these are time-ordered
    sensor readings; interpolating within each asset's own chronological
    sequence preserves the trend instead of flattening it. Any values that
    still can't be interpolated (e.g. leading/trailing gaps) fall back to the
    asset's median, then the global median as a last resort.
    """
    df = df.copy()
    numeric_cols = list(numeric_cols)
    if df[numeric_cols].isnull().sum().sum() == 0:
        return df

    def _impute_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(time_col)
        group[numeric_cols] = group[numeric_cols].interpolate(
            method="linear", limit_direction="both"
        )
        return group

    df = df.groupby(asset_col, group_keys=False).apply(_impute_group)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)


@dataclass
class LabelEncoders:
    """Holds fitted encoders so the same mapping can be reapplied at inference time."""

    encoders: dict[str, LabelEncoder] = field(default_factory=dict)

    def fit_transform(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            le = LabelEncoder()
            df[f"{col}_Encoded"] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        return df

    def transform(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            le = self.encoders[col]
            df[f"{col}_Encoded"] = le.transform(df[col].astype(str))
        return df


CLASS_NAMES = ["Normal", "Warning", "Critical", "Bearing_Failure", "Motor_Failure", "Seal_Failure"]
TERMINAL_STATES = {"Bearing_Failure", "Motor_Failure", "Seal_Failure"}


def map_failure_state_classes(df: pd.DataFrame, col: str = "Failure_State") -> pd.DataFrame:
    """Map Failure_State to a 6-class integer target, ordered by CLASS_NAMES.

    Normal=0, Warning=1, Critical=2, Bearing_Failure=3, Motor_Failure=4,
    Seal_Failure=5. All rows are kept and used for classification. Note the
    three terminal-failure classes are extremely rare and concentrated in a
    single asset (9 / 7 / 4 rows respectively, all from one pump) -- expect
    those specific classes to be hard or impossible to learn reliably; this
    is reported explicitly in the evaluation stage rather than hidden by
    macro-averaged metrics. `Is_Terminal_Failure` is kept as an informational
    flag (e.g. for RUL_Hours == 0 sanity checks), not for exclusion.
    """
    df = df.copy()
    class_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    df["Is_Terminal_Failure"] = df[col].isin(TERMINAL_STATES)
    df["Failure_Class"] = df[col].map(class_map)
    return df


class FeatureScaler:
    """Thin wrapper around StandardScaler that remembers which columns it scaled."""

    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.columns] = self.scaler.fit_transform(df[self.columns])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

ROLLING_WINDOWS = {"4h": 4, "12h": 12, "24h": 24}
LAG_STEPS = (1, 2, 3)


def add_rolling_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    asset_col: str = "Asset_ID",
    windows: dict[str, int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add rolling mean/std features per asset.

    NOTE ON WINDOW SEMANTICS: the raw data is sampled once every 20 hours per
    asset (20 pumps interleaved on an hourly log). A true calendar-time
    window ("last 4 hours") would almost always contain zero prior readings
    and be useless. The "4h / 12h / 24h" windows from the spec are therefore
    reinterpreted as the last 4 / 12 / 24 READINGS for that asset, which is
    the closest faithful equivalent and still captures short/medium/long-term
    trend at increasing look-back depth.
    """
    df = df.copy()
    grouped = df.groupby(asset_col)
    for col in cols:
        for label, window in windows.items():
            df[f"{col}_Rolling_Mean_{label}"] = grouped[col].transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
            )
            df[f"{col}_Rolling_Std_{label}"] = grouped[col].transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).std()
            )
    # First reading per asset has no variance to compute -> fill with 0 (no observed instability yet)
    std_cols = [c for c in df.columns if c.endswith(tuple(f"Std_{k}" for k in windows))]
    df[std_cols] = df[std_cols].fillna(0.0)
    return df


def add_lag_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    asset_col: str = "Asset_ID",
    lags: Iterable[int] = LAG_STEPS,
) -> pd.DataFrame:
    """Add lag_t-1 / t-2 / t-3 features per asset, back-filled at series starts."""
    df = df.copy()
    grouped = df.groupby(asset_col)
    for col in cols:
        for lag in lags:
            df[f"{col}_t-{lag}"] = grouped[col].shift(lag)
    lag_cols = [f"{c}_t-{l}" for c in cols for l in lags]
    df[lag_cols] = df.groupby(df[asset_col])[lag_cols].transform(
        lambda s: s.bfill()
    )
    return df


def add_rate_and_pct_change(
    df: pd.DataFrame, cols: Iterable[str], asset_col: str = "Asset_ID"
) -> pd.DataFrame:
    """Add first-difference rate of change and percentage change per asset.

    Rate of change flags how fast a sensor is moving (e.g. a sudden vibration
    spike), which is often a stronger failure precursor than the absolute
    level. Percentage change normalizes that movement relative to the
    reading's own scale, making spikes comparable across sensors with very
    different units (psi vs mm/s).
    """
    df = df.copy()
    grouped = df.groupby(asset_col)
    for col in cols:
        df[f"{col}_Rate_of_Change"] = grouped[col].diff().fillna(0.0)
        df[f"{col}_Pct_Change"] = (
            grouped[col].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically-motivated interaction terms between telemetry channels.

    Individual sensors can each look "normal" while their relationship drifts
    -- e.g. rising temperature at constant pressure can indicate friction/
    bearing wear that neither channel flags alone. Multiplicative interaction
    terms let tree-based models split on that joint behaviour directly
    instead of having to approximate it from two separate features.
    """
    df = df.copy()
    df["Temp_x_Pressure"] = df["Temperature_C"] * df["Pressure_psi"]
    df["Vibration_x_Temp"] = df["Vibration_mm_s"] * df["Temperature_C"]
    df["Vibration_x_Pressure"] = df["Vibration_mm_s"] * df["Pressure_psi"]
    df["Flow_x_Pressure"] = df["Flow_Rate_m3_h"] * df["Pressure_psi"]
    return df


def engineer_all_features(df: pd.DataFrame, asset_col: str = "Asset_ID") -> pd.DataFrame:
    """Run the full feature-engineering stage in the correct order."""
    df = add_rolling_features(df, RAW_NUMERIC_COLS, asset_col=asset_col)
    df = add_lag_features(df, RAW_NUMERIC_COLS, asset_col=asset_col)
    df = add_rate_and_pct_change(df, RAW_NUMERIC_COLS, asset_col=asset_col)
    df = add_interaction_features(df)
    return df


def remove_correlated_features(
    df: pd.DataFrame, columns: Iterable[str], threshold: float = 0.9
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedily drop one feature from each pair whose |correlation| exceeds threshold.

    For each correlated pair, the second column (in input order) is dropped
    and the first is kept, so callers can bias what survives by ordering
    `columns` with preferred/raw features first.
    """
    columns = list(columns)
    corr = df[columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop: list[str] = []
    dropped_pairs: list[tuple[str, str, float]] = []
    for col in upper.columns:
        if col in to_drop:
            continue
        high_corr = upper.index[(upper[col] > threshold) & (~upper.index.isin(to_drop))]
        for other in high_corr:
            if other == col or other in to_drop:
                continue
            to_drop.append(other)
            dropped_pairs.append((col, other, float(upper.loc[other, col])))
    kept = [c for c in columns if c not in to_drop]
    return kept, dropped_pairs


# ---------------------------------------------------------------------------
# 4. Data preparation / splitting
# ---------------------------------------------------------------------------


def group_aware_time_split(
    df: pd.DataFrame,
    asset_col: str = "Asset_ID",
    time_col: str = "Timestamp",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 80:20 chronologically WITHIN each asset, not randomly across rows.

    Why not a plain random 80:20 split: consecutive readings for the same
    pump are highly autocorrelated (rolling/lag features derive directly
    from neighboring rows), and RUL_Hours decays almost deterministically
    within a degradation cycle. A random row-level split would put a pump's
    hour-T reading in train and its near-identical hour-(T+1) neighbor in
    test, leaking future information and making test performance look far
    better than it would be in production. Instead, each asset's own
    timeline is sorted and the last `test_size` fraction of ITS readings is
    held out -- the model is always evaluated on genuinely future, unseen
    degradation behaviour for every pump.
    """
    train_parts, test_parts = [], []
    for _, group in df.sort_values(time_col).groupby(asset_col):
        n_test = max(1, int(len(group) * test_size))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])
    train_df = pd.concat(train_parts).sort_index()
    test_df = pd.concat(test_parts).sort_index()
    return train_df, test_df


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------


def evaluate_classifier(
    name: str, model, X_test, y_test, X_train=None, y_train=None, cv_folds: int = 5,
    labels: list[int] | None = None,
) -> dict:
    """Compute the standard classification metric suite for one fitted model.

    `labels` fixes the confusion matrix to a known, consistent set of class
    codes (e.g. `range(len(CLASS_NAMES))`) so every model's matrix has the
    same shape even if a rare class happens to be absent from a given split.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    try:
        y_proba = model.predict_proba(X_test)
        metrics["ROC_AUC"] = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro", labels=model.classes_
        )
    except (AttributeError, ValueError, IndexError):
        # Fires when a class present in training has zero examples in the test
        # split (expected here for the rarest failure-type classes) -- ROC-AUC
        # is undefined for a class with no positive samples, so this metric is
        # reported as NaN rather than crashing the run.
        metrics["ROC_AUC"] = np.nan

    if X_train is not None and y_train is not None:
        # The rarest classes (e.g. Seal_Failure, 4 rows total) can have fewer
        # training examples than cv_folds after the train/test split -- cap
        # n_splits at the smallest class count and skip CV entirely if even
        # 2-fold stratification isn't possible.
        min_class_count = int(pd.Series(y_train).value_counts().min())
        n_splits = min(cv_folds, min_class_count)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
                metrics["CV_F1_Mean"] = cv_scores.mean()
                metrics["CV_F1_Std"] = cv_scores.std()
            except ValueError:
                metrics["CV_F1_Mean"] = np.nan
                metrics["CV_F1_Std"] = np.nan
        else:
            metrics["CV_F1_Mean"] = np.nan
            metrics["CV_F1_Std"] = np.nan

    cm_labels = labels if labels is not None else sorted(
        set(pd.Series(y_test).unique()) | set(pd.Series(y_pred).unique())
    )
    metrics["_confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=cm_labels)
    metrics["_confusion_matrix_labels"] = cm_labels
    metrics["_y_pred"] = y_pred
    return metrics


def evaluate_regressor(
    name: str, model, X_test, y_test, X_train=None, y_train=None, cv_folds: int = 5
) -> dict:
    """Compute the standard regression metric suite for one fitted model."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": rmse,
        "R2_Score": r2_score(y_test, y_pred),
    }
    if X_train is not None and y_train is not None:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
        )
        metrics["CV_MAE_Mean"] = -cv_scores.mean()
        metrics["CV_MAE_Std"] = cv_scores.std()
    metrics["_y_pred"] = y_pred
    return metrics
