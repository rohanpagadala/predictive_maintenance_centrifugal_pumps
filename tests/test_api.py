def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["classifier_name"]
    assert body["regressor_name"]


def test_metrics_endpoint_shape(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("total_requests", "failed_requests", "prediction_count", "model_load_time_ms"):
        assert key in body


def test_predict_endpoint_success(client):
    payload = {"readings": [{
        "machine_model": "Nova-P", "location": "Onshore-Terminal-1", "vibration_mm_s": 2.0,
        "temperature_c": 62.0, "pressure_psi": 145.0, "flow_rate_m3_h": 250.0,
    }]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["failure_state"] == "Normal"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["remaining_useful_life"] >= 0.0
    assert "X-Process-Time-Ms" in resp.headers


def test_predict_endpoint_empty_readings_returns_422(client):
    resp = client.post("/predict", json={"readings": []})
    assert resp.status_code == 422


def test_predict_endpoint_bad_type_returns_422(client):
    payload = {"readings": [{"machine_model": "Nova-P", "vibration_mm_s": "not-a-number"}]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_endpoint_missing_machine_model_returns_422(client):
    payload = {"readings": [{"vibration_mm_s": 2.0, "temperature_c": 62.0}]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_endpoint_out_of_range_value_returns_422(client):
    payload = {"readings": [{"machine_model": "Nova-P", "vibration_mm_s": -5.0}]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_csv_endpoint_success(client):
    csv_content = (
        "Asset_ID,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        "PUMP_1,Nova-P,Onshore-Terminal-1,2.0,62,145,250\n"
        "PUMP_2,Titan-X3,Onshore-Terminal-1,4.8,80,133,236\n"
    )
    files = {"file": ("test.csv", csv_content, "text/csv")}
    resp = client.post("/predict-csv", files=files)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "PUMP_1" in resp.text and "PUMP_2" in resp.text
    assert "attachment; filename=predictions.csv" in resp.headers["content-disposition"]


def test_predict_csv_endpoint_rejects_non_csv_extension(client):
    files = {"file": ("test.txt", "not a csv", "text/plain")}
    resp = client.post("/predict-csv", files=files)
    assert resp.status_code == 422


def test_predict_csv_endpoint_missing_required_column_returns_422(client):
    files = {"file": ("test.csv", "Foo,Bar\n1,2\n", "text/csv")}
    resp = client.post("/predict-csv", files=files)
    assert resp.status_code == 422


def test_predict_csv_endpoint_rejects_oversized_upload(client, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    csv_content = (
        "Asset_ID,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        "PUMP_1,Nova-P,Onshore-Terminal-1,2.0,62,145,250\n"
    )
    files = {"file": ("test.csv", csv_content, "text/csv")}
    resp = client.post("/predict-csv", files=files)
    assert resp.status_code == 413


def test_explain_endpoint_returns_shap_explanation(client):
    payload = {"readings": [{
        "machine_model": "Nova-P", "location": "Onshore-Terminal-1", "vibration_mm_s": 2.0,
        "temperature_c": 62.0, "pressure_psi": 145.0, "flow_rate_m3_h": 250.0,
    }]}
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"]["failure_state"]
    assert body["predicted_class_label"] == body["prediction"]["failure_state"]
    assert len(body["classifier_top_features"]) > 0
    assert len(body["regressor_top_features"]) > 0
    assert body["classifier_waterfall_png_base64"]
    assert body["regressor_waterfall_png_base64"]


def test_metrics_prometheus_endpoint_returns_exposition_format(client):
    resp = client.get("/metrics/prometheus")
    assert resp.status_code == 200
    assert "pdm_requests_total" in resp.text
    assert "pdm_cpu_usage_percent" in resp.text


def test_drift_endpoint_returns_a_status(client):
    resp = client.get("/drift")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"ok", "unavailable"}


def test_audit_recent_returns_records_after_a_prediction(client):
    payload = {"readings": [{
        "asset_id": "PUMP_AUDIT_API_TEST", "machine_model": "Nova-P", "location": "Onshore-Terminal-1",
        "vibration_mm_s": 2.0, "temperature_c": 62.0, "pressure_psi": 145.0, "flow_rate_m3_h": 250.0,
    }]}
    predict_resp = client.post("/predict", json=payload)
    assert predict_resp.status_code == 200
    resp = client.get("/audit/asset/PUMP_AUDIT_API_TEST")
    assert resp.status_code == 200
    records = resp.json()["records"]
    assert len(records) >= 1
    assert records[0]["asset_id"] == "PUMP_AUDIT_API_TEST"
    assert records[0]["model_version"]

    import json
    logged_features = json.loads(records[0]["input_features"])
    assert logged_features.get("Location") == "Onshore-Terminal-1"


def test_models_endpoint_lists_active_version(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    versions = resp.json()["versions"]
    assert any(v["is_stable"] for v in versions)


def test_models_rollback_with_no_history_returns_409(client, tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    import model_registry as mr
    import pdm_utils as u
    from app import config

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
    )
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)

    resp = client.post("/models/rollback")
    assert resp.status_code == 409


def test_models_promote_unknown_version_returns_404(client):
    resp = client.post("/models/v999/promote")
    assert resp.status_code == 404


def test_predict_csv_endpoint_rejects_empty_csv_with_clean_error(client):
    files = {"file": ("test.csv", "Asset_ID,Machine_Model,Location\n", "text/csv")}
    resp = client.post("/predict-csv", files=files)
    assert resp.status_code == 422
    assert resp.json()["detail"]


def test_predict_csv_endpoint_scores_duplicate_asset_ids_independently(client):
    csv_content = (
        "Asset_ID,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        "PUMP_API_DUP,Nova-P,Onshore-Terminal-1,2.0,62,145,250\n"
        "PUMP_API_DUP,Nova-P,Onshore-Terminal-1,2.1,63,144,249\n"
    )
    files = {"file": ("test.csv", csv_content, "text/csv")}
    resp = client.post("/predict-csv", files=files)
    assert resp.status_code == 200
    assert resp.text.count("PUMP_API_DUP") == 2
