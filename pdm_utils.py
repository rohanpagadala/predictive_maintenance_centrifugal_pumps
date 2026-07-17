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

VALID_RANGES = {
    "Vibration_mm_s": (0, 15),
    "Temperature_C": (40, 130),
    "Pressure_psi": (80, 160),
    "Flow_Rate_m3_h": (140, 270),
}


def load_raw_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df


def parse_and_sort_timestamps(
    df: pd.DataFrame, asset_col: str = "Asset_ID", time_col: str = "Timestamp"
) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([asset_col, time_col]).reset_index(drop=True)
    return df


def drop_duplicates_and_report(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, n_before - len(df)


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    asset_col: str = "Asset_ID",
    time_col: str = "Timestamp",
) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = list(numeric_cols)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if df[numeric_cols].isnull().sum().sum() == 0:
        return df

    df = df.sort_values([asset_col, time_col]).reset_index(drop=True)
    df[numeric_cols] = df.groupby(asset_col)[numeric_cols].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


@dataclass
class LabelEncoders:

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


RAW_FAILURE_STATES = ["Normal", "Warning", "Critical", "Bearing_Failure", "Motor_Failure", "Seal_Failure"]
TERMINAL_STATES = {"Bearing_Failure", "Motor_Failure", "Seal_Failure"}
HEALTH_STATES = ["Normal", "Warning", "Critical"]
CLASS_NAMES = HEALTH_STATES

_HEALTH_STATE_MAP = {state: ("Critical" if state in TERMINAL_STATES else state) for state in RAW_FAILURE_STATES}


def map_failure_state_classes(df: pd.DataFrame, col: str = "Failure_State") -> pd.DataFrame:
    df = df.copy()
    class_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    df["Is_Terminal_Failure"] = df[col].isin(TERMINAL_STATES)
    df["Health_State"] = df[col].map(_HEALTH_STATE_MAP)
    df["Failure_Class"] = df["Health_State"].map(class_map)
    return df


def compute_fault_diagnosis_baseline(
    df: pd.DataFrame, raw_cols: Iterable[str] = RAW_NUMERIC_COLS,
    state_col: str = "Failure_State", baseline_state: str = "Normal",
) -> dict[str, dict[str, float]]:
    baseline_rows = df.loc[df[state_col] == baseline_state, list(raw_cols)]
    return {
        col: {"mean": float(baseline_rows[col].mean()), "std": float(baseline_rows[col].std())}
        for col in raw_cols
    }


class FeatureScaler:

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


ROLLING_WINDOWS = {"4h": 4, "12h": 12, "24h": 24}
LAG_STEPS = (1, 2, 3)


def add_rolling_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    asset_col: str = "Asset_ID",
    windows: dict[str, int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
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
    std_cols = [c for c in df.columns if c.endswith(tuple(f"Std_{k}" for k in windows))]
    df[std_cols] = df[std_cols].fillna(0.0)
    return df


def add_lag_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    asset_col: str = "Asset_ID",
    lags: Iterable[int] = LAG_STEPS,
) -> pd.DataFrame:
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
    df = df.copy()
    grouped = df.groupby(asset_col)
    for col in cols:
        df[f"{col}_Rate_of_Change"] = grouped[col].diff().fillna(0.0)
        df[f"{col}_Pct_Change"] = (
            grouped[col].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Temp_x_Pressure"] = df["Temperature_C"] * df["Pressure_psi"]
    df["Vibration_x_Temp"] = df["Vibration_mm_s"] * df["Temperature_C"]
    df["Vibration_x_Pressure"] = df["Vibration_mm_s"] * df["Pressure_psi"]
    df["Flow_x_Pressure"] = df["Flow_Rate_m3_h"] * df["Pressure_psi"]
    return df


def engineer_all_features(df: pd.DataFrame, asset_col: str = "Asset_ID") -> pd.DataFrame:
    df = add_rolling_features(df, RAW_NUMERIC_COLS, asset_col=asset_col)
    df = add_lag_features(df, RAW_NUMERIC_COLS, asset_col=asset_col)
    df = add_rate_and_pct_change(df, RAW_NUMERIC_COLS, asset_col=asset_col)
    df = add_interaction_features(df)
    return df


def remove_correlated_features(
    df: pd.DataFrame, columns: Iterable[str], threshold: float = 0.9
) -> tuple[list[str], list[tuple[str, str, float]]]:
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


def group_aware_time_split(
    df: pd.DataFrame,
    asset_col: str = "Asset_ID",
    time_col: str = "Timestamp",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    for _, group in df.sort_values(time_col).groupby(asset_col):
        n_test = max(1, int(len(group) * test_size))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])
    train_df = pd.concat(train_parts).sort_index()
    test_df = pd.concat(test_parts).sort_index()
    return train_df, test_df


def evaluate_classifier(
    name: str, model, X_test, y_test, X_train=None, y_train=None, cv_folds: int = 5,
    labels: list[int] | None = None,
) -> dict:
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
        metrics["ROC_AUC"] = np.nan

    if X_train is not None and y_train is not None:
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
