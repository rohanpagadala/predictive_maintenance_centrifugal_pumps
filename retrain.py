from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

import model_persistence as mp
import model_registry as mr
import pdm_utils as u
from app import feature_engineering

REQUIRED_TRAINING_COLUMNS = [
    "Timestamp", "Asset_ID", "Machine_Model", "Location",
    "Vibration_mm_s", "Temperature_C", "Pressure_psi", "Flow_Rate_m3_h",
    "RUL_Hours", "Failure_State",
]

TRAINING_STAGES = [
    "Loading Dataset",
    "Validating Data",
    "Preprocessing",
    "Feature Engineering",
    "Training Health Classification Model",
    "Training Remaining Useful Life Model",
    "Evaluating Models",
    "Saving Models",
]

_CLASSIFIER_FACTORY: dict[str, Callable[[], object]] = {
    "LightGBM": lambda: LGBMClassifier(random_state=u.RANDOM_STATE, class_weight="balanced", verbose=-1),
    "XGBoost": lambda: XGBClassifier(random_state=u.RANDOM_STATE, eval_metric="mlogloss"),
    "Random Forest": lambda: RandomForestClassifier(random_state=u.RANDOM_STATE, class_weight="balanced"),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=u.RANDOM_STATE, class_weight="balanced"),
}

_REGRESSOR_FACTORY: dict[str, Callable[[], object]] = {
    "Random Forest": lambda: RandomForestRegressor(random_state=u.RANDOM_STATE),
    "XGBoost": lambda: XGBRegressor(random_state=u.RANDOM_STATE),
    "LightGBM": lambda: LGBMRegressor(random_state=u.RANDOM_STATE, verbose=-1),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=u.RANDOM_STATE),
}


def _build_classifier(name: str):
    return _CLASSIFIER_FACTORY.get(name, _CLASSIFIER_FACTORY["LightGBM"])()


def _build_regressor(name: str):
    return _REGRESSOR_FACTORY.get(name, _REGRESSOR_FACTORY["Random Forest"])()


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    n_missing: int = 0
    n_duplicates: int = 0
    preview: pd.DataFrame | None = None


def validate_training_dataset(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if len(df) == 0:
        errors.append("Dataset has no rows.")
        return ValidationResult(
            is_valid=False, errors=errors, warnings=warnings,
            n_rows=0, n_cols=len(df.columns), n_missing=0, n_duplicates=0, preview=df.head(10),
        )

    missing_cols = [c for c in REQUIRED_TRAINING_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required column(s): {', '.join(missing_cols)}")

    n_missing = int(df.isnull().sum().sum())
    if n_missing:
        warnings.append(f"{n_missing} missing value(s) found across the dataset (will be imputed).")

    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        warnings.append(f"{n_duplicates} duplicate row(s) found (will be dropped).")

    if not missing_cols:
        for col in [*u.RAW_NUMERIC_COLS, "RUL_Hours"]:
            coerced = pd.to_numeric(df[col], errors="coerce")
            invalid = coerced.isnull() & df[col].notnull()
            if invalid.any():
                errors.append(f"Column '{col}' contains non-numeric value(s).")

        unknown_states = set(df["Failure_State"].astype(str).unique()) - set(u.RAW_FAILURE_STATES)
        if unknown_states:
            errors.append(f"Unrecognized Failure_State value(s): {', '.join(sorted(unknown_states))}")

    return ValidationResult(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        n_rows=len(df),
        n_cols=len(df.columns),
        n_missing=n_missing,
        n_duplicates=n_duplicates,
        preview=df.head(10),
    )


@dataclass
class RetrainResult:
    version: str
    n_training_rows: int
    n_test_rows: int
    new_classifier_metrics: dict
    new_regressor_metrics: dict
    current_classifier_metrics: dict
    current_regressor_metrics: dict
    artifacts: mp.ModelArtifacts
    warnings: list[str] = field(default_factory=list)


def retrain_pipeline(
    df_raw: pd.DataFrame,
    current_artifacts: mp.ModelArtifacts,
    models_root: str | Path,
    repo_dir: str | Path,
    dataset_version: str,
    on_stage: Callable[[str], None] | None = None,
) -> RetrainResult:

    def stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    stage("Loading Dataset")
    df = df_raw.copy()

    stage("Validating Data")
    validation = validate_training_dataset(df)
    if not validation.is_valid:
        raise ValueError("Dataset failed validation: " + "; ".join(validation.errors))

    stage("Preprocessing")
    df = u.parse_and_sort_timestamps(df)
    df, _ = u.drop_duplicates_and_report(df)
    df = u.handle_missing_values(df, u.RAW_NUMERIC_COLS)
    categorical_cols = current_artifacts.feature_engineering_config["categorical_cols"]
    encoders = u.LabelEncoders()
    df = encoders.fit_transform(df, categorical_cols)
    df = u.map_failure_state_classes(df)
    fault_diagnosis_baseline = u.compute_fault_diagnosis_baseline(df)

    stage("Feature Engineering")
    feature_list = list(current_artifacts.feature_list)
    df_fe = feature_engineering.engineer(df, current_artifacts.feature_engineering_config, feature_list)

    train_df, test_df = u.group_aware_time_split(df_fe)
    scaler = u.FeatureScaler(feature_list)
    scaler.fit_transform(train_df[feature_list])

    X_train, X_test = train_df[feature_list], test_df[feature_list]
    y_train_clf, y_test_clf = train_df["Failure_Class"], test_df["Failure_Class"]
    y_train_reg, y_test_reg = train_df["RUL_Hours"], test_df["RUL_Hours"]

    stage("Training Health Classification Model")
    classifier = _build_classifier(current_artifacts.classifier_name)
    if isinstance(classifier, XGBClassifier):
        classifier.fit(X_train, y_train_clf, sample_weight=compute_sample_weight("balanced", y_train_clf))
    else:
        classifier.fit(X_train, y_train_clf)

    stage("Training Remaining Useful Life Model")
    regressor = _build_regressor(current_artifacts.regressor_name)
    regressor.fit(X_train, y_train_reg)

    stage("Evaluating Models")
    new_classifier_metrics = u.evaluate_classifier(
        current_artifacts.classifier_name, classifier, X_test, y_test_clf, X_train, y_train_clf,
    )
    new_regressor_metrics = u.evaluate_regressor(
        current_artifacts.regressor_name, regressor, X_test, y_test_reg, X_train, y_train_reg,
    )
    current_classifier_metrics = u.evaluate_classifier(
        current_artifacts.classifier_name, current_artifacts.classifier, X_test, y_test_clf,
    )
    current_regressor_metrics = u.evaluate_regressor(
        current_artifacts.regressor_name, current_artifacts.regressor, X_test, y_test_reg,
    )

    stage("Saving Models")
    version = mr.register_version(
        classifier=classifier, classifier_name=current_artifacts.classifier_name,
        regressor=regressor, regressor_name=current_artifacts.regressor_name,
        scaler=scaler, encoders=encoders,
        feature_list=feature_list, class_names=current_artifacts.class_names,
        feature_engineering_config=current_artifacts.feature_engineering_config,
        models_root=models_root,
        dataset_version=dataset_version,
        performance_metrics={
            "classifier_test_accuracy": round(float(new_classifier_metrics["Accuracy"]), 4),
            "classifier_test_precision_macro": round(float(new_classifier_metrics["Precision"]), 4),
            "classifier_test_recall_macro": round(float(new_classifier_metrics["Recall"]), 4),
            "classifier_test_f1_macro": round(float(new_classifier_metrics["F1_Score"]), 4),
            "regressor_test_mae_hours": round(float(new_regressor_metrics["MAE"]), 4),
            "regressor_test_rmse_hours": round(float(new_regressor_metrics["RMSE"]), 4),
            "regressor_test_r2": round(float(new_regressor_metrics["R2_Score"]), 4),
            "n_training_rows": len(train_df),
        },
        repo_dir=repo_dir,
        promote_to_stable=False,
        fault_diagnosis_baseline=fault_diagnosis_baseline,
    )

    retrain_warnings: list[str] = []
    import joblib

    from app import drift_detection as dd
    from app import explainability as xai

    version_dir = Path(models_root) / version
    try:
        drift_reference = dd.build_reference_from_raw_data(df)
        dd.save_reference(drift_reference, version_dir / dd.REFERENCE_FILENAME)

        background_sample = df_fe[feature_list].sample(n=min(500, len(df_fe)), random_state=u.RANDOM_STATE)
        joblib.dump(background_sample, version_dir / xai.BACKGROUND_SAMPLE_FILENAME)
    except Exception as exc:
        retrain_warnings.append(
            f"Registered version {version}, but could not build its drift-reference/SHAP-background "
            f"artifacts ({exc}). /drift and /explain will be degraded for this version until it's rebuilt."
        )

    new_artifacts = mr.load_models_from_registry(models_root, version=version)
    return RetrainResult(
        version=version,
        n_training_rows=len(train_df),
        n_test_rows=len(test_df),
        new_classifier_metrics=new_classifier_metrics,
        new_regressor_metrics=new_regressor_metrics,
        current_classifier_metrics=current_classifier_metrics,
        current_regressor_metrics=current_regressor_metrics,
        artifacts=new_artifacts,
        warnings=retrain_warnings,
    )


def recommend_replacement(result: RetrainResult) -> bool:
    new_f1 = result.new_classifier_metrics["F1_Score"]
    cur_f1 = result.current_classifier_metrics["F1_Score"]
    new_rmse = result.new_regressor_metrics["RMSE"]
    cur_rmse = result.current_regressor_metrics["RMSE"]
    at_least_as_good = new_f1 >= cur_f1 and new_rmse <= cur_rmse
    strictly_better = new_f1 > cur_f1 or new_rmse < cur_rmse
    return at_least_as_good and strictly_better


def training_history(models_root: str | Path) -> pd.DataFrame:
    rows = []
    for v in mr.list_versions(models_root):
        rows.append({
            "Model Version": v["version"],
            "Training Date": v.get("training_date", ""),
            "Dataset Size": v.get("n_training_rows"),
            "Classification Accuracy": v.get("classifier_test_accuracy", v.get("classifier_test_recall_macro")),
            "Regression RMSE": v.get("regressor_test_rmse_hours"),
            "Status": "Current" if v["is_stable"] else "Previous",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_sort_key"] = pd.to_numeric(
        df["Model Version"].str.extract(r"(\d+)")[0], errors="coerce"
    ).fillna(-1)
    return df.sort_values("_sort_key", ascending=False).drop(columns="_sort_key").reset_index(drop=True)
