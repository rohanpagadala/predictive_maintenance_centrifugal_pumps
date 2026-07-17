import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import model_persistence as mp
import pdm_utils as u


@pytest.fixture
def dummy_artifacts(tmp_path):
    X = pd.DataFrame(np.random.RandomState(0).rand(30, 3), columns=["a", "b", "c"])
    y_class = np.random.RandomState(0).randint(0, 2, 30)
    y_reg = np.random.RandomState(0).rand(30) * 100

    clf = DecisionTreeClassifier(random_state=0).fit(X, y_class)
    reg = DecisionTreeRegressor(random_state=0).fit(X, y_reg)
    scaler = u.FeatureScaler(["a", "b", "c"])
    scaler.fit_transform(X)
    encoders = u.LabelEncoders()
    encoders.fit_transform(pd.DataFrame({"m": ["x", "y", "x"]}), ["m"])

    models_dir = tmp_path / "models"
    mp.save_models(
        classifier=clf, classifier_name="Decision Tree (Baseline)",
        regressor=reg, regressor_name="Decision Tree (Baseline)",
        scaler=scaler, encoders=encoders,
        feature_list=["a", "b", "c"], class_names=["c0", "c1"],
        feature_engineering_config={"raw_numeric_cols": ["a", "b", "c"]},
        models_dir=models_dir,
    )
    return models_dir, X, clf, reg


def test_save_models_writes_all_expected_files(dummy_artifacts):
    models_dir, *_ = dummy_artifacts
    expected = {
        "classification_model_decision_tree_baseline.joblib",
        "regression_model_decision_tree_baseline.joblib",
        "feature_scaler.joblib", "label_encoders.joblib",
        "selected_features.joblib", "class_names.joblib",
        "feature_engineering_config.joblib", "fault_diagnosis_baseline.json", "metadata.json",
    }
    actual = {p.name for p in models_dir.iterdir()}
    assert expected <= actual


def test_load_models_round_trip_matches_original_predictions(dummy_artifacts):
    models_dir, X, clf, reg = dummy_artifacts
    artifacts = mp.load_models(models_dir)

    assert artifacts.classifier_name == "Decision Tree (Baseline)"
    assert artifacts.feature_list == ["a", "b", "c"]

    np.testing.assert_array_equal(artifacts.classifier.predict(X), clf.predict(X))
    np.testing.assert_allclose(artifacts.regressor.predict(X), reg.predict(X))


def test_load_models_missing_directory_raises_persistence_error(tmp_path):
    with pytest.raises(mp.ModelPersistenceError):
        mp.load_models(tmp_path / "does_not_exist")


def test_predict_helper_reorders_columns_and_labels_classes(dummy_artifacts):
    models_dir, X, clf, reg = dummy_artifacts
    artifacts = mp.load_models(models_dir)
    shuffled = X[["c", "a", "b"]]
    result = mp.predict(artifacts, shuffled)
    assert list(result["Failure_Class_Predicted"]) == list(clf.predict(X))
    assert set(result["Failure_State_Predicted"]) <= {"c0", "c1"}


def test_predict_helper_raises_on_missing_columns(dummy_artifacts):
    models_dir, X, *_ = dummy_artifacts
    artifacts = mp.load_models(models_dir)
    with pytest.raises(mp.ModelPersistenceError):
        mp.predict(artifacts, X[["a", "b"]])
