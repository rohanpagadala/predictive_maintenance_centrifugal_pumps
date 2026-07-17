from pathlib import Path

import pandas as pd
import pytest

import model_registry as mr
import retrain

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "AI_Predictive_Maintenance_Pump_Dataset_10000.xlsx"


@pytest.fixture(scope="module")
def small_training_dataset() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    asset_ids = sorted(df["Asset_ID"].unique())[:4]
    return df[df["Asset_ID"].isin(asset_ids)].reset_index(drop=True)


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
    import numpy as np
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
