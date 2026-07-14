"""
Shared utilities: logging setup, timing helpers, and the exception types
every other app module raises so `api.py` can map them to HTTP status codes
in one place.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Any, Iterator

import numpy as np

from app import config

_CONFIGURED = False


def setup_logging() -> None:
    """Configure root logging once: console + rotating file handlers.

    Safe to call multiple times (e.g. once per Streamlit rerun) -- only
    attaches handlers on the first call.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(config.LOG_LEVEL)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        config.API_LOG_FILE, maxBytes=5_000_000, backupCount=3
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    error_handler = RotatingFileHandler(
        config.ERROR_LOG_FILE, maxBytes=5_000_000, backupCount=3
    )
    error_handler.setFormatter(fmt)
    error_handler.setLevel(logging.WARNING)
    root.addHandler(error_handler)

    _CONFIGURED = True


def get_prediction_logger() -> logging.Logger:
    """A dedicated logger for prediction results, written to its own file
    so prediction audit trails don't get lost in general API chatter."""
    logger = logging.getLogger("pdm.predictions")
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(config.PREDICTION_LOG_FILE)
               for h in logger.handlers):
        handler = RotatingFileHandler(config.PREDICTION_LOG_FILE, maxBytes=5_000_000, backupCount=3)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[dict]:
    """Context manager that logs and returns elapsed wall-clock time in ms.

    Usage:
        with timed(logger, "model load") as t:
            ...
        # t["elapsed_ms"] is available after the block exits
    """
    start = time.perf_counter()
    result: dict[str, float] = {}
    try:
        yield result
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        result["elapsed_ms"] = elapsed_ms
        logger.info("%s took %.2f ms", label, elapsed_ms)


def to_json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalar types to plain Python types for JSON
    serialization -- FastAPI's default encoder chokes on np.float32/np.int64."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class PdMError(Exception):
    """Base class for all handled errors in the serving layer."""


class InputValidationError(PdMError):
    """Raised when request data fails validation beyond what Pydantic checks
    (e.g. a fully-empty history, an unusable timestamp ordering)."""


class InferenceError(PdMError):
    """Raised when the loaded models fail to produce a prediction."""


class ArtifactLoadError(PdMError):
    """Raised when saved model artifacts can't be loaded at startup."""
