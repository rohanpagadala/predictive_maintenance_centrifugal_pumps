"""Tests for the FastAPI endpoints -- HTTP-level: status codes, response
shape, and error mapping. Pipeline correctness itself is covered by
test_predict.py; these tests focus on what changes at the HTTP boundary.
"""


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
        "machine_model": "Nova-P", "vibration_mm_s": 2.0, "temperature_c": 62.0,
        "pressure_psi": 145.0, "flow_rate_m3_h": 250.0,
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
        "Asset_ID,Machine_Model,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        "PUMP_1,Nova-P,2.0,62,145,250\n"
        "PUMP_2,Titan-X3,4.8,80,133,236\n"
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
