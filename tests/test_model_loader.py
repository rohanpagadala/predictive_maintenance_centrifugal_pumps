"""Tests for app.model_loader: artifacts load correctly and are cached."""

from app import model_loader


def test_artifacts_have_expected_shape(artifacts):
    assert artifacts.classifier_name
    assert artifacts.regressor_name
    assert len(artifacts.feature_list) > 0
    assert artifacts.class_names == [
        "Normal", "Warning", "Critical", "Bearing_Failure", "Motor_Failure", "Seal_Failure",
    ]
    assert "raw_numeric_cols" in artifacts.feature_engineering_config


def test_models_are_fitted_and_predict_capable(artifacts):
    assert hasattr(artifacts.classifier, "predict")
    assert hasattr(artifacts.classifier, "predict_proba")
    assert hasattr(artifacts.regressor, "predict")


def test_get_artifacts_is_cached_not_reloaded():
    first = model_loader.get_artifacts()
    second = model_loader.get_artifacts()
    assert first is second  # same object -> no redundant disk read


def test_force_reload_returns_equivalent_but_fresh_object():
    first = model_loader.get_artifacts()
    second = model_loader.get_artifacts(force_reload=True)
    assert first is not second
    assert first.classifier_name == second.classifier_name
