import pandas as pd

from app import fault_diagnosis as fd

_BASELINE = {
    "Vibration_mm_s": {"mean": 2.0, "std": 0.46},
    "Temperature_C": {"mean": 62.47, "std": 4.32},
    "Pressure_psi": {"mean": 144.99, "std": 2.89},
    "Flow_Rate_m3_h": {"mean": 250.0, "std": 2.89},
}


def _row(**overrides):
    base = {"Vibration_mm_s": 2.0, "Temperature_C": 62.5, "Pressure_psi": 145.0, "Flow_Rate_m3_h": 250.0}
    base.update(overrides)
    return pd.Series(base)


def test_dominant_vibration_diagnoses_bearing_failure():
    result = fd.diagnose(_row(**{"Vibration_mm_s": 15.0}), _BASELINE)
    assert result.predicted_fault == fd.FAULT_BEARING
    assert result.dominant_channel == "Vibration_mm_s"


def test_dominant_temperature_diagnoses_motor_failure():
    result = fd.diagnose(_row(**{"Temperature_C": 110.0}), _BASELINE)
    assert result.predicted_fault == fd.FAULT_MOTOR
    assert result.dominant_channel == "Temperature_C"


def test_dominant_pressure_and_flow_drop_diagnoses_seal_failure():
    result = fd.diagnose(_row(**{"Pressure_psi": 100.0, "Flow_Rate_m3_h": 180.0}), _BASELINE)
    assert result.predicted_fault == fd.FAULT_SEAL
    assert "Pressure_psi" in result.dominant_channel and "Flow_Rate_m3_h" in result.dominant_channel


def test_ambiguous_reading_diagnoses_unknown_critical_fault():
    result = fd.diagnose(_row(**{
        "Vibration_mm_s": 10.0, "Temperature_C": 65.0, "Pressure_psi": 95.0, "Flow_Rate_m3_h": 200.0,
    }), _BASELINE)
    assert result.predicted_fault == fd.FAULT_UNKNOWN
    assert result.dominant_channel == "none (ambiguous)"


def test_result_confidence_is_bounded():
    result = fd.diagnose(_row(**{"Vibration_mm_s": 15.0}), _BASELINE)
    assert 0.0 <= result.confidence <= 1.0
