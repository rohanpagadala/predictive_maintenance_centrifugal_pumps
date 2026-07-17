import numpy as np
import pandas as pd
import pytest

import model_registry as mr
import retrain

SYNTHETIC_MACHINE_MODELS = ["Nova-P", "Titan-X3"]
SYNTHETIC_LOCATIONS = ["Onshore-Terminal-1", "Onshore-Terminal-2"]


def _build_synthetic_training_dataframe(n_assets: int = 4, rows_per_asset: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    base_time = pd.Timestamp("2026-01-01")
    rows = []
    for a in range(n_assets):
        asset_id = f"PUMP_{a:04d}"
        for i in range(rows_per_asset):
            is_warning = i >= rows_per_asset - 5
            rows.append({
                "Timestamp": base_time + pd.Timedelta(hours=i),
                "Asset_ID": asset_id,
                "Machine_Model": SYNTHETIC_MACHINE_MODELS[a % len(SYNTHETIC_MACHINE_MODELS)],
                "Location": SYNTHETIC_LOCATIONS[a % len(SYNTHETIC_LOCATIONS)],
                "Vibration_mm_s": round(rng.uniform(4.0, 6.0) if is_warning else rng.uniform(1.5, 3.0), 2),
                "Temperature_C": round(rng.uniform(75, 85) if is_warning else rng.uniform(55, 68), 2),
                "Pressure_psi": round(rng.uniform(130, 138) if is_warning else rng.uniform(140, 150), 2),
                "Flow_Rate_m3_h": round(rng.uniform(220, 235) if is_warning else rng.uniform(240, 260), 2),
                "RUL_Hours": round(rng.uniform(20, 50) if is_warning else rng.uniform(200, 500), 2),
                "Failure_State": "Warning" if is_warning else "Normal",
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def small_training_dataset(tmp_path_factory) -> pd.DataFrame:
    df = _build_synthetic_training_dataframe()
    tmp_dir = tmp_path_factory.mktemp("retrain_test_data")
    xlsx_path = tmp_dir / "synthetic_training_dataset.xlsx"
    df.to_excel(xlsx_path, index=False)
    return pd.read_excel(xlsx_path)


def test_validate_training_dataset_rejects_empty_dataframe():
    empty = pd.DataFrame(columns=retrain.REQUIRED_TRAINING_COLUMNS)
    result = retrain.validate_training_dataset(empty)
    assert result.is_valid is False
    assert any("no rows" in e.lower() for e in result.errors)


def test_validate_training_dataset_accepts_well_formed_data(small_training_dataset):
    result = retrain.validate_training_dataset(small_training_dataset)
    assert result.is_valid is True
    assert result.n_rows == len(small_training_dataset)


def test_validate_training_dataset_rejects_missing_columns():
    result = retrain.validate_training_dataset(pd.DataFrame({"Foo": [1, 2]}))
    assert result.is_valid is False
    assert any("Missing required column" in e for e in result.errors)


def test_training_history_tolerates_non_numeric_version(tmp_path):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    import pdm_utils as u

    X = pd.DataFrame(np.random.RandomState(0).rand(20, 2), columns=["a", "b"])
    y_class = np.random.RandomState(0).randint(0, 2, 20)
    y_reg = np.random.RandomState(0).rand(20) * 100
    clf = DecisionTreeClassifier(random_state=0).fit(X, y_class)
    reg = DecisionTreeRegressor(random_state=0).fit(X, y_reg)
    scaler = u.FeatureScaler(["a", "b"])
    scaler.fit_transform(X)
    encoders = u.LabelEncoders()
    encoders.fit_transform(pd.DataFrame({"m": ["x", "y"]}), ["m"])

    mr.register_version(
        classifier=clf, classifier_name="DT", regressor=reg, regressor_name="DT",
        scaler=scaler, encoders=encoders, feature_list=["a", "b"], class_names=["c0", "c1"],
        feature_engineering_config={}, models_root=tmp_path,
        dataset_version="test.xlsx", performance_metrics={"acc": 0.9}, repo_dir=".",
        version="baseline", promote_to_stable=True,
    )

    history_df = retrain.training_history(tmp_path)
    assert list(history_df["Model Version"]) == ["baseline"]


def test_retrain_pipeline_end_to_end_registers_a_new_version(small_training_dataset, artifacts, tmp_path):
    result = retrain.retrain_pipeline(
        small_training_dataset,
        current_artifacts=artifacts,
        models_root=tmp_path,
        repo_dir=tmp_path,
        dataset_version="test_subset.xlsx",
    )
    assert result.version
    assert result.n_training_rows > 0
    assert result.n_test_rows > 0
    assert "F1_Score" in result.new_classifier_metrics
    assert "RMSE" in result.new_regressor_metrics

    versions = mr.list_versions(tmp_path)
    assert any(v["version"] == result.version for v in versions)

    assert result.warnings == []
