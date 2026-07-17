import pandas as pd
import pytest

from app import fault_diagnosis
from app.predict import apply_rul_consistency_rule, predict, predict_batch
from app.preprocessing import dataframe_from_csv_bytes
from app.schemas import PredictionRequest, TelemetryReading
from app.utils import InputValidationError


def _reading(**overrides) -> TelemetryReading:
    base = dict(
        machine_model="Nova-P", location="Onshore-Terminal-1", vibration_mm_s=2.0,
        temperature_c=62.0, pressure_psi=145.0, flow_rate_m3_h=250.0,
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


def test_normal_reading_never_gets_fault_diagnosis():
    normal_resp = predict(PredictionRequest(readings=[_reading()]))
    assert normal_resp.fault_diagnosis is None


def test_critical_reading_gets_fault_diagnosis():
    critical_resp = predict(PredictionRequest(readings=[
        _reading(vibration_mm_s=8.0, temperature_c=98.0, pressure_psi=118.0, flow_rate_m3_h=205.0)
    ]))
    assert critical_resp.failure_state == "Critical"
    assert critical_resp.fault_diagnosis is not None
    assert critical_resp.fault_diagnosis.predicted_fault in (
        fault_diagnosis.SPECIFIC_FAULTS | {fault_diagnosis.FAULT_UNKNOWN}
    )
    assert 0.0 <= critical_resp.remaining_useful_life <= 10.0


def test_warning_reading_also_gets_fault_diagnosis():
    warning_resp = predict(PredictionRequest(readings=[
        _reading(vibration_mm_s=4.5, temperature_c=80.0, pressure_psi=135.0, flow_rate_m3_h=235.0)
    ]))
    assert warning_resp.failure_state == "Warning"
    assert warning_resp.fault_diagnosis is not None
    assert warning_resp.fault_diagnosis.predicted_fault in (
        fault_diagnosis.SPECIFIC_FAULTS | {fault_diagnosis.FAULT_UNKNOWN}
    )


def test_apply_rul_consistency_rule_zeroes_rul_only_for_critical_specific_faults():
    states = pd.Series(["Normal", "Critical", "Critical", "Warning"])
    rul = pd.Series([300.0, 8.0, 7.0, 30.0])
    diagnosed = pd.Series([None, fault_diagnosis.FAULT_BEARING, fault_diagnosis.FAULT_UNKNOWN, fault_diagnosis.FAULT_BEARING])
    result = apply_rul_consistency_rule(states, rul, diagnosed)
    assert result.tolist() == [300.0, 0.0, 7.0, 30.0]


def test_class_probabilities_sum_to_one():
    resp = predict(PredictionRequest(readings=[_reading()]))
    assert abs(sum(resp.class_probabilities.values()) - 1.0) < 1e-3


def test_unknown_machine_model_produces_warning_not_crash():
    resp = predict(PredictionRequest(readings=[_reading(machine_model="NotARealModel-9000")]))
    assert len(resp.warnings) == 1
    assert "Unknown Machine_Model" in resp.warnings[0]
    assert resp.failure_state


def test_missing_channel_with_no_history_raises_input_validation_error():
    with pytest.raises(InputValidationError):
        predict(PredictionRequest(readings=[_reading(pressure_psi=None)]))


def test_missing_channel_with_history_is_imputed_successfully():
    readings = [_reading(asset_id="PUMP_HIST", pressure_psi=145.0 + i * 0.1) for i in range(5)]
    readings.append(_reading(asset_id="PUMP_HIST", pressure_psi=None))
    resp = predict(PredictionRequest(readings=readings))
    assert resp.failure_state


def test_all_channels_missing_raises_pydantic_validation_error():
    with pytest.raises(Exception):
        PredictionRequest(readings=[_reading(
            vibration_mm_s=None, temperature_c=None, pressure_psi=None, flow_rate_m3_h=None,
        )])


def test_predict_batch_preserves_row_count():
    csv_bytes = (
        b"Asset_ID,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        b"PUMP_X,Nova-P,Onshore-Terminal-1,2.0,62,145,250\n"
        b"PUMP_Y,Titan-X3,Onshore-Terminal-1,4.8,80,133,236\n"
        b"PUMP_Z,Atlas-7,Onshore-Terminal-1,8.0,98,100,200\n"
    )
    df = dataframe_from_csv_bytes(csv_bytes)
    out = predict_batch(df)
    assert len(out) == 3
    assert {"Asset_ID", "Failure_State_Predicted", "Confidence", "RUL_Hours_Predicted"} <= set(out.columns)


def test_predict_batch_multi_row_asset_history_grouped_correctly():
    csv_bytes = (
        b"Asset_ID,Timestamp,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        b"PUMP_A,2026-01-01 00:00,Nova-P,Onshore-Terminal-1,2.0,60,145,250\n"
        b"PUMP_A,2026-01-01 20:00,Nova-P,Onshore-Terminal-1,2.1,61,144,249\n"
        b"PUMP_B,2026-01-01 00:00,Titan-X3,Onshore-Terminal-1,5.0,82,132,235\n"
    )
    df = dataframe_from_csv_bytes(csv_bytes)
    out = predict_batch(df)
    assert len(out) == 3
    assert (out["Asset_ID"] == "PUMP_A").sum() == 2
    assert (out["Asset_ID"] == "PUMP_B").sum() == 1


def test_predict_batch_diagnoses_warning_and_critical_rows_but_never_normal():
    csv_bytes = (
        b"Asset_ID,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        b"PUMP_NORMAL,Nova-P,Onshore-Terminal-1,2.0,62,145,250\n"
        b"PUMP_WARN,Nova-P,Onshore-Terminal-1,4.5,80,135,235\n"
        b"PUMP_CRIT,Atlas-7,Onshore-Terminal-1,8.0,98,118,205\n"
    )
    df = dataframe_from_csv_bytes(csv_bytes)
    out = predict_batch(df)
    assert "Diagnosed_Fault" in out.columns
    normal_row = out[out["Asset_ID"] == "PUMP_NORMAL"].iloc[0]
    warn_row = out[out["Asset_ID"] == "PUMP_WARN"].iloc[0]
    crit_row = out[out["Asset_ID"] == "PUMP_CRIT"].iloc[0]
    assert normal_row["Diagnosed_Fault"] == ""
    if warn_row["Failure_State_Predicted"] == "Warning":
        assert warn_row["Diagnosed_Fault"] != ""
    if crit_row["Failure_State_Predicted"] == "Critical":
        assert crit_row["Diagnosed_Fault"] != ""


def test_dataframe_from_csv_bytes_rejects_missing_machine_model_column():
    with pytest.raises(InputValidationError):
        dataframe_from_csv_bytes(b"Foo,Bar\n1,2\n")


def test_dataframe_from_csv_bytes_rejects_empty_csv():
    with pytest.raises(InputValidationError):
        dataframe_from_csv_bytes(b"Asset_ID,Machine_Model\n")


def test_dataframe_from_csv_bytes_rejects_null_asset_id():
    csv_bytes = (
        b"Asset_ID,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        b",Nova-P,Onshore-Terminal-1,2.0,62,145,250\n"
    )
    with pytest.raises(InputValidationError):
        dataframe_from_csv_bytes(csv_bytes)


def test_single_and_batch_predict_agree_for_the_same_reading():
    reading = _reading(asset_id="PUMP_EQUIV", vibration_mm_s=2.0, temperature_c=62.0, pressure_psi=145.0, flow_rate_m3_h=250.0)
    single_resp = predict(PredictionRequest(readings=[reading]))

    csv_bytes = (
        b"Asset_ID,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        b"PUMP_EQUIV,Nova-P,Onshore-Terminal-1,2.0,62,145,250\n"
    )
    batch_out = predict_batch(dataframe_from_csv_bytes(csv_bytes))
    batch_row = batch_out.iloc[0]

    assert batch_row["Failure_State_Predicted"] == single_resp.failure_state
    assert batch_row["Confidence"] == pytest.approx(single_resp.confidence, abs=1e-4)
    assert batch_row["RUL_Hours_Predicted"] == pytest.approx(single_resp.remaining_useful_life, abs=1e-2)


def test_predict_batch_scores_duplicate_asset_ids_independently():
    csv_bytes = (
        b"Asset_ID,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        b"PUMP_DUP,Nova-P,Onshore-Terminal-1,2.0,62,145,250\n"
        b"PUMP_DUP,Nova-P,Onshore-Terminal-1,2.1,63,144,249\n"
        b"PUMP_DUP,Nova-P,Onshore-Terminal-1,2.2,64,143,248\n"
    )
    out = predict_batch(dataframe_from_csv_bytes(csv_bytes))
    assert len(out) == 3
    assert (out["Asset_ID"] == "PUMP_DUP").sum() == 3
