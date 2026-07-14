"""Tests for the app.predict inference pipeline: single + batch prediction,
missing values, and unknown categories -- called directly (no HTTP layer),
so these isolate pipeline correctness from API/serialization concerns
(that's what test_api.py is for).
"""

import pytest

from app.predict import predict, predict_batch
from app.preprocessing import dataframe_from_csv_bytes
from app.schemas import PredictionRequest, TelemetryReading
from app.utils import InputValidationError


def _reading(**overrides) -> TelemetryReading:
    base = dict(
        machine_model="Nova-P", vibration_mm_s=2.0, temperature_c=62.0,
        pressure_psi=145.0, flow_rate_m3_h=250.0,
    )
    base.update(overrides)
    return TelemetryReading(**base)


def test_predict_normal_reading_returns_normal():
    resp = predict(PredictionRequest(readings=[_reading()]))
    assert resp.failure_state == "Normal"
    assert 0.0 <= resp.confidence <= 1.0
    assert resp.remaining_useful_life >= 0.0


def test_predict_warning_range_reading_returns_warning_or_worse():
    resp = predict(PredictionRequest(readings=[
        _reading(vibration_mm_s=4.5, temperature_c=80.0, pressure_psi=135.0, flow_rate_m3_h=235.0)
    ]))
    assert resp.failure_state in {"Warning", "Critical"}


def test_class_probabilities_sum_to_one():
    resp = predict(PredictionRequest(readings=[_reading()]))
    assert abs(sum(resp.class_probabilities.values()) - 1.0) < 1e-3


def test_unknown_machine_model_produces_warning_not_crash():
    resp = predict(PredictionRequest(readings=[_reading(machine_model="NotARealModel-9000")]))
    assert len(resp.warnings) == 1
    assert "Unknown machine_model" in resp.warnings[0]
    assert resp.failure_state  # still produced a prediction


def test_missing_channel_with_no_history_raises_input_validation_error():
    # A lone reading with a fully-missing channel has nothing to interpolate
    # from and no batch to take a median over -- this must fail loudly
    # rather than silently feed NaN into the model.
    with pytest.raises(InputValidationError):
        predict(PredictionRequest(readings=[_reading(pressure_psi=None)]))


def test_missing_channel_with_history_is_imputed_successfully():
    # The same missing channel, but with prior readings for the asset to
    # interpolate from, should succeed.
    readings = [_reading(asset_id="PUMP_HIST", pressure_psi=145.0 + i * 0.1) for i in range(5)]
    readings.append(_reading(asset_id="PUMP_HIST", pressure_psi=None))
    resp = predict(PredictionRequest(readings=readings))
    assert resp.failure_state


def test_all_channels_missing_raises_pydantic_validation_error():
    with pytest.raises(Exception):  # pydantic ValidationError, raised by TelemetryReading/PredictionRequest itself
        PredictionRequest(readings=[_reading(
            vibration_mm_s=None, temperature_c=None, pressure_psi=None, flow_rate_m3_h=None,
        )])


def test_predict_batch_preserves_row_count():
    csv_bytes = (
        b"Asset_ID,Machine_Model,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        b"PUMP_X,Nova-P,2.0,62,145,250\n"
        b"PUMP_Y,Titan-X3,4.8,80,133,236\n"
        b"PUMP_Z,Atlas-7,8.0,98,100,200\n"
    )
    df = dataframe_from_csv_bytes(csv_bytes)
    out = predict_batch(df)
    assert len(out) == 3
    assert {"Asset_ID", "Failure_State_Predicted", "Confidence", "RUL_Hours_Predicted"} <= set(out.columns)


def test_predict_batch_multi_row_asset_history_grouped_correctly():
    csv_bytes = (
        b"Asset_ID,Timestamp,Machine_Model,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        b"PUMP_A,2026-01-01 00:00,Nova-P,2.0,60,145,250\n"
        b"PUMP_A,2026-01-01 20:00,Nova-P,2.1,61,144,249\n"
        b"PUMP_B,2026-01-01 00:00,Titan-X3,5.0,82,132,235\n"
    )
    df = dataframe_from_csv_bytes(csv_bytes)
    out = predict_batch(df)
    assert len(out) == 3
    assert (out["Asset_ID"] == "PUMP_A").sum() == 2
    assert (out["Asset_ID"] == "PUMP_B").sum() == 1


def test_dataframe_from_csv_bytes_rejects_missing_machine_model_column():
    with pytest.raises(InputValidationError):
        dataframe_from_csv_bytes(b"Foo,Bar\n1,2\n")


def test_dataframe_from_csv_bytes_rejects_empty_csv():
    with pytest.raises(InputValidationError):
        dataframe_from_csv_bytes(b"Asset_ID,Machine_Model\n")
