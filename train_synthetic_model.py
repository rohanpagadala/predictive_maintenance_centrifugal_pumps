
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42

NUMERIC_FEATURES = ["Vibration", "Temperature", "Pressure", "Flow_Rate", "Operating_Hours"]
CATEGORICAL_FEATURES = ["Machine_Model", "Location"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

HEALTH_ORDER = ["Normal", "Warning", "Critical"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = FEATURE_COLUMNS + ["Health_Status", "Fault_Type", "Remaining_Useful_Life_Hours"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}")
    return df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le
    return train_df, test_df, encoders


def evaluate_classifier(name: str, model, X_test, y_test, labels: list[str]) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "Model": name,
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return {**metrics, "_report": report, "_confusion_matrix": cm, "_labels": labels, "_y_pred": y_pred}


def evaluate_regressor(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(root_mean_squared_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
        "_y_pred": y_pred,
    }


def save_confusion_matrix_plot(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.55), max(5, len(labels) * 0.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=str(BASE_DIR / "data" / "synthetic_pump_dataset.csv"))
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "models" / "synthetic_v1"))
    parser.add_argument("--figures-dir", type=str, default=str(BASE_DIR / "outputs" / "figures"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = load_data(Path(args.data))
    print(f"Loaded {len(df):,} rows")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=RANDOM_STATE, stratify=df["Fault_Type"]
    )
    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    train_enc, test_enc, encoders = encode_categoricals(train_df, test_df)
    X_train, X_test = train_enc[FEATURE_COLUMNS], test_enc[FEATURE_COLUMNS]

    fault_labels = sorted(df["Fault_Type"].unique())

    print("\nTraining Health_Status classifier (LightGBM) ...")
    y_train_health, y_test_health = train_df["Health_Status"], test_df["Health_Status"]
    health_clf = LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)
    health_clf.fit(X_train, y_train_health, categorical_feature=CATEGORICAL_FEATURES)
    health_metrics = evaluate_classifier("Health_Status", health_clf, X_test, y_test_health, HEALTH_ORDER)

    print("Training Fault_Type classifier (LightGBM) ...")
    y_train_fault, y_test_fault = train_df["Fault_Type"], test_df["Fault_Type"]
    fault_clf = LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)
    fault_clf.fit(X_train, y_train_fault, categorical_feature=CATEGORICAL_FEATURES)
    fault_metrics = evaluate_classifier("Fault_Type", fault_clf, X_test, y_test_fault, fault_labels)

    print("Training Remaining_Useful_Life_Hours regressor (XGBoost) ...")
    y_train_rul, y_test_rul = train_df["Remaining_Useful_Life_Hours"], test_df["Remaining_Useful_Life_Hours"]
    rul_reg = XGBRegressor(random_state=RANDOM_STATE, n_estimators=400, max_depth=6, learning_rate=0.05)
    rul_reg.fit(X_train, y_train_rul)
    rul_metrics = evaluate_regressor("Remaining_Useful_Life_Hours", rul_reg, X_test, y_test_rul)

    rul_pred_by_bucket = pd.DataFrame({
        "Health_Status": y_test_health.values,
        "y_true": y_test_rul.values,
        "y_pred": rul_metrics["_y_pred"],
    })
    rul_pred_by_bucket["abs_err"] = (rul_pred_by_bucket["y_true"] - rul_pred_by_bucket["y_pred"]).abs()
    rul_by_bucket = rul_pred_by_bucket.groupby("Health_Status")["abs_err"].agg(["mean", "median", "count"]).reindex(HEALTH_ORDER)

    print("\n" + "=" * 78)
    print("HEALTH STATUS CLASSIFIER")
    print("=" * 78)
    print(f"Accuracy: {health_metrics['Accuracy']:.4f}  |  Macro F1: {health_metrics['F1_macro']:.4f}  |  "
          f"Macro Precision: {health_metrics['Precision_macro']:.4f}  |  Macro Recall: {health_metrics['Recall_macro']:.4f}")
    print(health_metrics["_report"])

    print("=" * 78)
    print("FAULT TYPE CLASSIFIER (15-way)")
    print("=" * 78)
    print(f"Accuracy: {fault_metrics['Accuracy']:.4f}  |  Macro F1: {fault_metrics['F1_macro']:.4f}  |  "
          f"Macro Precision: {fault_metrics['Precision_macro']:.4f}  |  Macro Recall: {fault_metrics['Recall_macro']:.4f}")
    print(fault_metrics["_report"])

    print("=" * 78)
    print("REMAINING USEFUL LIFE REGRESSOR")
    print("=" * 78)
    print(f"MAE: {rul_metrics['MAE']:.1f} hrs  |  RMSE: {rul_metrics['RMSE']:.1f} hrs  |  R2: {rul_metrics['R2']:.4f}")
    print("\nMean absolute error by true Health_Status bucket:")
    print(rul_by_bucket.round(1))

    print("\n" + "=" * 78)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 78)
    for name, model in [("Health_Status", health_clf), ("Fault_Type", fault_clf)]:
        imp = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        print(f"\n{name}:")
        print(imp.to_string())

    save_confusion_matrix_plot(
        health_metrics["_confusion_matrix"], HEALTH_ORDER,
        "Health_Status - Confusion Matrix", figures_dir / "synthetic_health_status_confusion_matrix.png",
    )
    save_confusion_matrix_plot(
        fault_metrics["_confusion_matrix"], fault_labels,
        "Fault_Type - Confusion Matrix", figures_dir / "synthetic_fault_type_confusion_matrix.png",
    )
    print(f"\nSaved confusion matrix plots to {figures_dir}/")

    joblib.dump(health_clf, out_dir / "health_status_classifier_lightgbm.joblib")
    joblib.dump(fault_clf, out_dir / "fault_type_classifier_lightgbm.joblib")
    joblib.dump(rul_reg, out_dir / "rul_regressor_xgboost.joblib")
    joblib.dump(encoders, out_dir / "label_encoders.joblib")
    joblib.dump(FEATURE_COLUMNS, out_dir / "feature_list.joblib")

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.data),
        "n_rows_total": len(df),
        "n_rows_train": len(train_df),
        "n_rows_test": len(test_df),
        "feature_columns": FEATURE_COLUMNS,
        "health_status_classes": HEALTH_ORDER,
        "fault_type_classes": fault_labels,
        "health_status_metrics": {k: v for k, v in health_metrics.items() if not k.startswith("_")},
        "fault_type_metrics": {k: v for k, v in fault_metrics.items() if not k.startswith("_")},
        "rul_metrics": {k: v for k, v in rul_metrics.items() if not k.startswith("_")},
        "rul_mae_by_health_status": rul_by_bucket["mean"].round(2).to_dict(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved models, encoders, and metadata.json to {out_dir}/")


if __name__ == "__main__":
    main()
