"""
Pydantic request/response models for the FastAPI layer.

These are the API's contract: FastAPI validates every request against
`TelemetryReading`/`PredictionRequest` before a single line of inference code
runs, and Swagger UI (`/docs`) renders them automatically.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TelemetryReading(BaseModel):
    """One raw sensor reading for one pump at one point in time.

    Telemetry fields are optional individually (a real sensor feed can drop
    a channel) but at least one must be present -- see `PredictionRequest`'s
    validator, which checks this across the whole submitted history.
    """

    timestamp: Optional[datetime] = Field(
        default=None, description="Reading time. If omitted, readings are assumed to be in submitted order."
    )
    asset_id: Optional[str] = Field(default=None, description="Pump identifier, e.g. 'PUMP_001'.")
    machine_model: str = Field(..., description="Pump model/design, e.g. 'Nova-P'.")
    vibration_mm_s: Optional[float] = Field(default=None, ge=0, le=100)
    temperature_c: Optional[float] = Field(default=None, ge=-50, le=300)
    pressure_psi: Optional[float] = Field(default=None, ge=0, le=1000)
    flow_rate_m3_h: Optional[float] = Field(default=None, ge=0, le=2000)

    @field_validator("machine_model")
    @classmethod
    def _non_empty_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("machine_model must not be empty")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2026-07-14T08:00:00",
                "asset_id": "PUMP_001",
                "machine_model": "Nova-P",
                "vibration_mm_s": 4.8,
                "temperature_c": 74.0,
                "pressure_psi": 132.0,
                "flow_rate_m3_h": 238.0,
            }
        }
    }


class PredictionRequest(BaseModel):
    """A chronological history of readings for a single pump.

    A single reading (list of length 1) is accepted -- rolling/lag features
    then degrade gracefully to that one point (documented in `app/predict.py`)
    -- but supplying the pump's last ~24 readings, if available, produces
    materially more reliable rolling/lag features and is recommended.
    """

    readings: list[TelemetryReading] = Field(..., min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _at_least_one_channel_per_reading(self) -> "PredictionRequest":
        for i, r in enumerate(self.readings):
            if all(
                v is None
                for v in (r.vibration_mm_s, r.temperature_c, r.pressure_psi, r.flow_rate_m3_h)
            ):
                raise ValueError(f"reading[{i}] has no telemetry values at all (all four channels are null)")
        return self


class PredictionResponse(BaseModel):
    """Prediction for the most recent reading in a submitted history."""

    asset_id: str
    timestamp: str
    failure_state: str = Field(..., description="Predicted class label, e.g. 'Warning'.")
    failure_class: int = Field(..., description="Predicted class as an integer 0-5, see class_names ordering.")
    confidence: float = Field(..., description="Classifier's predicted probability for failure_state.")
    remaining_useful_life: float = Field(..., description="Predicted RUL in hours.")
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list, description="Non-fatal notices, e.g. unknown machine_model.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classifier_name: Optional[str] = None
    regressor_name: Optional[str] = None
    n_features: Optional[int] = None
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
