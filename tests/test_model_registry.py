import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import model_registry as mr
import pdm_utils as u


def _dummy_registered(tmp_path, version=None, promote=False):
    X = pd.DataFrame(np.random.RandomState(0).rand(20, 2), columns=["a", "b"])
    y_class = np.random.RandomState(0).randint(0, 2, 20)
    y_reg = np.random.RandomState(0).rand(20) * 100
    clf = DecisionTreeClassifier(random_state=0).fit(X, y_class)
    reg = DecisionTreeRegressor(random_state=0).fit(X, y_reg)
    scaler = u.FeatureScaler(["a", "b"])
    scaler.fit_transform(X)
    encoders = u.LabelEncoders()
    encoders.fit_transform(pd.DataFrame({"m": ["x", "y"]}), ["m"])

    return mr.register_version(
        classifier=clf, classifier_name="DT", regressor=reg, regressor_name="DT",
        scaler=scaler, encoders=encoders, feature_list=["a", "b"], class_names=["c0", "c1"],
        feature_engineering_config={}, models_root=tmp_path,
        dataset_version="test.xlsx", performance_metrics={"acc": 0.9},
        repo_dir=".", version=version, promote_to_stable=promote,
    )


def test_first_registered_version_becomes_stable_automatically(tmp_path):
    v1 = _dummy_registered(tmp_path)
    assert mr.get_active_version(tmp_path) == v1 == "v1"


def test_second_version_is_latest_but_not_automatically_stable(tmp_path):
    v1 = _dummy_registered(tmp_path)
    v2 = _dummy_registered(tmp_path)
    assert v2 == "v2"
    versions = {v["version"]: v for v in mr.list_versions(tmp_path)}
    assert versions[v1]["is_stable"] is True
    assert versions[v2]["is_stable"] is False
    assert versions[v2]["is_latest"] is True
    assert mr.get_active_version(tmp_path) == v1


def test_promote_to_stable_switches_active_version(tmp_path):
    _dummy_registered(tmp_path)
    v2 = _dummy_registered(tmp_path)
    mr.promote_to_stable(tmp_path, v2)
    assert mr.get_active_version(tmp_path) == v2


def test_rollback_reverts_to_previous_stable(tmp_path):
    v1 = _dummy_registered(tmp_path)
    v2 = _dummy_registered(tmp_path)
    mr.promote_to_stable(tmp_path, v2)
    reverted_to = mr.rollback_to_previous(tmp_path)
    assert reverted_to == v1
    assert mr.get_active_version(tmp_path) == v1


def test_rollback_with_no_history_raises(tmp_path):
    _dummy_registered(tmp_path)
    with pytest.raises(mr.ModelRegistryError):
        mr.rollback_to_previous(tmp_path)


def test_promote_unknown_version_raises(tmp_path):
    _dummy_registered(tmp_path)
    with pytest.raises(mr.ModelRegistryError):
        mr.promote_to_stable(tmp_path, "v99")


def test_metadata_includes_provenance_fields(tmp_path):
    _dummy_registered(tmp_path)
    versions = mr.list_versions(tmp_path)
    meta = versions[0]
    for field in ("model_version", "training_date", "dataset_version", "git_commit_hash", "acc"):
        assert field in meta


def test_load_models_from_registry_matches_original_predictions(tmp_path):
    v1 = _dummy_registered(tmp_path)
    artifacts = mr.load_models_from_registry(tmp_path, version=v1)
    assert artifacts.classifier_name == "DT"
    assert artifacts.feature_list == ["a", "b"]


def test_promote_to_version_with_missing_files_raises_and_leaves_registry_untouched(tmp_path):
    v1 = _dummy_registered(tmp_path)
    v2 = _dummy_registered(tmp_path)
    (tmp_path / v2 / "feature_scaler.joblib").unlink()

    with pytest.raises(mr.ModelRegistryError, match="missing artifact file"):
        mr.promote_to_stable(tmp_path, v2)
    assert mr.get_active_version(tmp_path) == v1


def test_rollback_to_version_with_missing_files_raises_and_leaves_registry_untouched(tmp_path):
    v1 = _dummy_registered(tmp_path)
    (tmp_path / v1 / "feature_scaler.joblib").unlink()
    v2 = _dummy_registered(tmp_path)
    mr.promote_to_stable(tmp_path, v2)

    with pytest.raises(mr.ModelRegistryError, match="missing artifact file"):
        mr.rollback_to_previous(tmp_path)
    assert mr.get_active_version(tmp_path) == v2
