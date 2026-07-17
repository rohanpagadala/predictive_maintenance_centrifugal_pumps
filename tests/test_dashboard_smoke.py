from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from app.api import app as fastapi_app

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "streamlit_app.py")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_api_url():
    port = _free_port()
    config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Test API server did not start in time.")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def at(live_api_url, monkeypatch):
    monkeypatch.setenv("PDM_API_URL", live_api_url)
    app_test = AppTest.from_file(APP_PATH, default_timeout=60)
    return app_test


def _goto(app_test: AppTest, page: str) -> AppTest:
    app_test.run()
    app_test.sidebar.radio[0].set_value(page).run()
    return app_test


def test_home_page_loads_without_exception(at):
    at.run()
    assert not at.exception


def test_overview_page_loads_without_exception(at):
    _goto(at, "Project Overview")
    assert not at.exception


def test_single_prediction_page_loads_and_scores_without_exception(at):
    _goto(at, "Single Prediction")
    assert not at.exception


def test_batch_prediction_page_loads_without_exception(at):
    _goto(at, "Batch Prediction")
    assert not at.exception


def test_batch_prediction_valid_csv_runs_without_exception(at):
    _goto(at, "Batch Prediction")
    csv_bytes = (
        b"Asset_ID,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
        b"PUMP_SMOKE_1,Nova-P,Onshore-Terminal-1,2.0,62,145,250\n"
        b"PUMP_SMOKE_2,Titan-X3,Onshore-Terminal-1,4.8,80,133,236\n"
    )
    at.file_uploader[0].upload("batch.csv", csv_bytes, "text/csv").run()
    assert not at.exception
    run_buttons = [b for b in at.button if b.label == "Run Batch Prediction"]
    assert run_buttons, "Run Batch Prediction button not found"
    run_buttons[0].click().run()
    assert not at.exception
    assert any("Scored" in md.value for md in at.success)


def test_batch_prediction_empty_csv_shows_clean_error_not_crash(at):
    _goto(at, "Batch Prediction")
    csv_bytes = b"Asset_ID,Machine_Model,Location,Vibration_mm_s,Temperature_C,Pressure_psi,Flow_Rate_m3_h\n"
    at.file_uploader[0].upload("empty.csv", csv_bytes, "text/csv").run()
    run_buttons = [b for b in at.button if b.label == "Run Batch Prediction"]
    run_buttons[0].click().run()
    assert not at.exception
    assert len(at.error) > 0


def test_batch_prediction_malformed_file_shows_clean_error_not_crash(at):
    _goto(at, "Batch Prediction")
    at.file_uploader[0].upload("garbage.csv", b"\x00\x01\xff not really a csv \xfe", "text/csv").run()
    assert not at.exception
    assert len(at.error) > 0


def test_prediction_history_page_loads_without_exception(at):
    _goto(at, "Prediction History")
    assert not at.exception


def test_prediction_history_filters_render_and_apply_without_exception(at):
    _goto(at, "Single Prediction")
    assert not at.exception

    _goto(at, "Prediction History")
    assert not at.exception
    selectboxes = [sb for sb in at.selectbox if sb.label in ("Machine Model", "Location", "Health Status")]
    assert len(selectboxes) == 3
    health_status_box = next(sb for sb in at.selectbox if sb.label == "Health Status")
    if len(health_status_box.options) > 1:
        health_status_box.set_value(health_status_box.options[1]).run()
    assert not at.exception


def test_about_page_loads_without_exception(at):
    _goto(at, "About")
    assert not at.exception
