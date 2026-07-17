import asyncio

import pytest
from fastapi import HTTPException

from app import config, security


def _call(header_value: str) -> None:
    asyncio.run(security.require_api_key(x_api_key=header_value))


def test_no_key_required_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "API_KEY_REQUIRED", False)
    _call("")


def test_valid_key_accepted_when_required(monkeypatch):
    monkeypatch.setattr(config, "API_KEY_REQUIRED", True)
    monkeypatch.setattr(config, "API_KEY", "correct-key")
    _call("correct-key")


def test_missing_key_rejected_when_required(monkeypatch):
    monkeypatch.setattr(config, "API_KEY_REQUIRED", True)
    monkeypatch.setattr(config, "API_KEY", "correct-key")
    with pytest.raises(HTTPException) as exc_info:
        _call("")
    assert exc_info.value.status_code == 401


def test_wrong_key_rejected_when_required(monkeypatch):
    monkeypatch.setattr(config, "API_KEY_REQUIRED", True)
    monkeypatch.setattr(config, "API_KEY", "correct-key")
    with pytest.raises(HTTPException) as exc_info:
        _call("wrong-key")
    assert exc_info.value.status_code == 401
