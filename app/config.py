from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ENVIRONMENT = os.getenv("PDM_ENV", "development").lower()
if ENVIRONMENT not in {"development", "testing", "production"}:
    raise ValueError(
        f"PDM_ENV={ENVIRONMENT!r} is not one of 'development', 'testing', 'production'."
    )
IS_DEVELOPMENT = ENVIRONMENT == "development"
IS_TESTING = ENVIRONMENT == "testing"
IS_PRODUCTION = ENVIRONMENT == "production"

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / f".env.{ENVIRONMENT}")
    load_dotenv(BASE_DIR / ".env", override=True)
except ImportError:
    pass


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    return [v.strip() for v in raw.split(",") if v.strip()] if raw else default


MODELS_DIR = Path(os.getenv("PDM_MODELS_DIR", BASE_DIR / "models"))
LOGS_DIR = Path(os.getenv("PDM_LOGS_DIR", BASE_DIR / "logs"))
OUTPUTS_DIR = Path(os.getenv("PDM_OUTPUTS_DIR", BASE_DIR / "outputs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

APP_NAME = "Predictive Maintenance API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "Failure-severity classification and remaining-useful-life regression "
    "for industrial centrifugal pumps."
)

_DEFAULT_LOG_LEVEL = {"development": "DEBUG", "testing": "INFO", "production": "WARNING"}[ENVIRONMENT]
LOG_LEVEL = os.getenv("PDM_LOG_LEVEL", _DEFAULT_LOG_LEVEL).upper()
API_LOG_FILE = LOGS_DIR / "api.log"
PREDICTION_LOG_FILE = LOGS_DIR / "predictions.log"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"

HOST = os.getenv("PDM_HOST", "0.0.0.0")
PORT = int(os.getenv("PDM_PORT", "8000"))

MIN_RECOMMENDED_HISTORY = 24

MAX_UPLOAD_BYTES = int(os.getenv("PDM_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

API_KEY_REQUIRED = _env_bool("PDM_API_KEY_REQUIRED", default=IS_PRODUCTION)
API_KEY = os.getenv("PDM_API_KEY", "")
if API_KEY_REQUIRED and not API_KEY:
    raise ValueError(
        "PDM_API_KEY_REQUIRED is true but PDM_API_KEY is not set -- refusing to start "
        "with auth enabled and no key configured (would lock out every caller)."
    )

_DEFAULT_CORS_ORIGINS = ["*"] if not IS_PRODUCTION else []
CORS_ALLOWED_ORIGINS = _env_list("PDM_ALLOWED_ORIGINS", default=_DEFAULT_CORS_ORIGINS)

RATE_LIMIT_PREDICT = os.getenv("PDM_RATE_LIMIT_PREDICT", "20/minute" if IS_PRODUCTION else "1000/minute")
RATE_LIMIT_DEFAULT = os.getenv("PDM_RATE_LIMIT_DEFAULT", "120/minute" if IS_PRODUCTION else "2000/minute")

VERBOSE_ERRORS = not IS_PRODUCTION
