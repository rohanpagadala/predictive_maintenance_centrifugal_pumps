"""
Central configuration for the PdM serving layer.

Every path and tunable lives here, sourced from environment variables (with
sane defaults) so the same code runs unmodified in a notebook-adjacent local
checkout, a Docker container, or a cloud deployment -- only the environment
changes, never the code.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional in production, where env vars are set directly

BASE_DIR = Path(__file__).resolve().parent.parent

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

LOG_LEVEL = os.getenv("PDM_LOG_LEVEL", "INFO").upper()
API_LOG_FILE = LOGS_DIR / "api.log"
PREDICTION_LOG_FILE = LOGS_DIR / "predictions.log"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"

HOST = os.getenv("PDM_HOST", "0.0.0.0")
PORT = int(os.getenv("PDM_PORT", "8000"))

# Minimum history rows recommended for high-fidelity rolling/lag features.
# Requests with fewer rows still work (pdm_utils degrades gracefully to
# single-point rolling stats) but are flagged in the response as lower
# confidence in the engineered-feature sense, not the model's own confidence.
MIN_RECOMMENDED_HISTORY = 24
