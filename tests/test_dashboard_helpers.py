from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))
import streamlit_app as sa


def test_estimated_replacement_date_is_after_now_and_formatted():
    result = sa.estimated_replacement_date(24.0)
    assert result.endswith("IST")
    assert sa.estimated_replacement_date(float("nan")) == ""


def test_diagnosis_parts_never_returns_raw_technical_fault_code():
    for health_state in ("Normal", "Warning", "Critical"):
        for fault in (None, "", "Unknown_Critical_Fault", "Bearing_Failure", "Motor_Failure", "Seal_Failure"):
            parsed = sa._diagnosis_parts(health_state, fault)
            assert parsed is not None
            label, color, name, description, action = parsed
            assert "_" not in name
            assert name != ""


def test_diagnosis_parts_critical_unknown_fault_has_friendly_name_and_action():
    label, color, name, description, action = sa._diagnosis_parts("Critical", "Unknown_Critical_Fault")
    assert name == "Unknown Critical Fault"
    assert description
    assert action


def test_diagnosis_parts_unknown_health_state_returns_none():
    assert sa._diagnosis_parts("SomethingElse", None) is None


def test_add_diagnosis_export_columns_produces_three_separate_columns():
    df = pd.DataFrame({
        "Failure_State_Predicted": ["Normal", "Warning", "Critical"],
        "Diagnosed_Fault": ["", "Motor_Failure", "Seal_Failure"],
    })
    out = sa.add_diagnosis_export_columns(df)
    assert {"Likely_Failure_Or_Developing_Issue", "Failure_Description", "Recommended_Action"} <= set(out.columns)
    assert out.loc[0, "Likely_Failure_Or_Developing_Issue"] == "Normal Operation"
    assert out.loc[1, "Likely_Failure_Or_Developing_Issue"] == "Motor Stress"
    assert out.loc[2, "Likely_Failure_Or_Developing_Issue"] == "Seal Failure"
    assert out.loc[2, "Recommended_Action"]


def test_build_batch_export_dataframe_includes_all_requested_columns():
    df = pd.DataFrame({
        "Asset_ID": ["PUMP_1", "PUMP_2"],
        "Timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "Failure_State_Predicted": ["Normal", "Critical"],
        "Confidence": [0.98, 0.87],
        "RUL_Hours_Predicted": [480.0, 8.0],
        "Diagnosed_Fault": ["", "Bearing_Failure"],
    })
    out = sa.build_batch_export_dataframe(df)
    expected = {
        "Priority", "Likely_Failure_Or_Developing_Issue", "Failure_Description",
        "Recommended_Action", "Estimated_Replacement_Date", "Model_Version",
    }
    assert expected <= set(out.columns)
    assert len(out) == 2
    assert list(out["Asset_ID"]) == ["PUMP_1", "PUMP_2"]


def test_rul_severity_thresholds_match_documented_bounds():
    assert sa.rul_severity(24.0) == "critical"
    assert sa.rul_severity(24.01) == "warning"
    assert sa.rul_severity(360.0) == "warning"
    assert sa.rul_severity(360.01) == "normal"
    assert sa.rul_severity(float("nan")) == "unknown"


def test_get_active_model_version_falls_back_to_unknown_when_api_unreachable(monkeypatch):
    monkeypatch.setattr(sa, "API_URL", "http://127.0.0.1:1")
    sa.get_active_model_version.clear()
    assert sa.get_active_model_version() == "unknown"


def test_cap_for_onscreen_display_leaves_small_batches_untouched():
    df = pd.DataFrame({"RUL_Hours_Predicted": [10.0, 20.0, 30.0]})
    capped, notice = sa.cap_for_onscreen_display(df)
    assert len(capped) == 3
    assert notice == ""


def test_cap_for_onscreen_display_caps_large_batches_to_most_urgent_rows():
    n = sa.MAX_ONSCREEN_TABLE_ROWS + 250
    df = pd.DataFrame({
        "Asset_ID": [f"PUMP_{i}" for i in range(n)],
        "RUL_Hours_Predicted": list(range(n, 0, -1)),
    })
    capped, notice = sa.cap_for_onscreen_display(df)
    assert len(capped) == sa.MAX_ONSCREEN_TABLE_ROWS
    assert notice != ""
    assert str(n) in notice
    assert capped["RUL_Hours_Predicted"].max() <= df["RUL_Hours_Predicted"].sort_values().iloc[sa.MAX_ONSCREEN_TABLE_ROWS - 1]
