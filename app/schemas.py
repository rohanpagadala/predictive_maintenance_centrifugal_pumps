from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

import pdm_utils as u

_VIBRATION_RANGE = u.VALID_RANGES["Vibration_mm_s"]
_TEMPERATURE_RANGE = u.VALID_RANGES["Temperature_C"]
_PRESSURE_RANGE = u.VALID_RANGES["Pressure_psi"]
_FLOW_RANGE = u.VALID_RANGES["Flow_Rate_m3_h"]


class TelemetryReading(BaseModel):

    timestamp: Optional[datetime] = Field(
        default=None, description="Reading time. If omitted, readings are assumed to be in submitted order."
    )
    asset_id: Optional[str] = Field(default=None, description="Pump identifier, e.g. 'PUMP_001'.")
    machine_model: str = Field(..., description="Pump model/design, e.g. 'Nova-P'.")
    location: str = Field(..., description="Site/platform, e.g. 'Onshore-Terminal-1'.")
    vibration_mm_s: Optional[float] = Field(
        default=None, ge=_VIBRATION_RANGE[0], le=_VIBRATION_RANGE[1],
        description="Training data observed range: 1.2-12.4 mm/s.",
    )
    temperature_c: Optional[float] = Field(
        default=None, ge=_TEMPERATURE_RANGE[0], le=_TEMPERATURE_RANGE[1],
        description="Training data observed range: 55.0-119.3 C.",
    )
    pressure_psi: Optional[float] = Field(
        default=None, ge=_PRESSURE_RANGE[0], le=_PRESSURE_RANGE[1],
        description="Training data observed range: 90.6-150.0 psi.",
    )
    flow_rate_m3_h: Optional[float] = Field(
        default=None, ge=_FLOW_RANGE[0], le=_FLOW_RANGE[1],
        description="Training data observed range: 160.0-255.0 m3/h.",
    )

    @field_validator("machine_model", "location")
    @classmethod
    def _non_empty_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("machine_model and location must not be empty")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2026-07-14T08:00:00",
                "asset_id": "PUMP_001",
                "machine_model": "Nova-P",
                "location": "Onshore-Terminal-1",
                "vibration_mm_s": 4.8,
                "temperature_c": 74.0,
                "pressure_psi": 132.0,
                "flow_rate_m3_h": 238.0,
            }
        }
    }


class PredictionRequest(BaseModel):

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


class FaultDiagnosis(BaseModel):

    predicted_fault: str = Field(
        ..., description="One of Bearing_Failure, Motor_Failure, Seal_Failure, Unknown_Critical_Fault."
    )
    confidence: float = Field(
        ..., description="Rule-engine confidence (0-1): how much the leading channel's deviation "
                          "leads the runner-up, not a calibrated probability."
    )
    dominant_channel: str = Field(..., description="Sensor channel(s) whose deviation from the Normal "
                                                     "baseline drove this diagnosis.")
    channel_deviation_scores: dict[str, float] = Field(
        ..., description="Per-channel |z-score| deviation from the Normal-state baseline."
    )
    rationale: str = Field(..., description="Human-readable explanation of the diagnosis.")


class PredictionResponse(BaseModel):

    asset_id: str
    timestamp: str
    failure_state: str = Field(..., description="Predicted health stage: Normal, Warning, or Critical.")
    failure_class: int = Field(..., description="Predicted class as an integer 0-2, see class_names ordering.")
    confidence: float = Field(..., description="Classifier's predicted probability for failure_state.")
    remaining_useful_life: float = Field(..., description="Predicted RUL in hours.")
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list, description="Non-fatal notices, e.g. unknown machine_model.")
    fault_diagnosis: Optional[FaultDiagnosis] = Field(
        default=None,
        description="Populated when failure_state is 'Warning' or 'Critical' (never for 'Normal') -- "
                    "see app/fault_diagnosis.py.",
    )


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: Optional[float] = None
    direction: str = Field(..., description="Which way this feature pushed the prediction -- label is context-specific, see /explain docs.")


class ExplanationResponse(BaseModel):

    prediction: PredictionResponse
    predicted_class_label: str
    classifier_top_features: list[FeatureContribution]
    regressor_top_features: list[FeatureContribution]
    classifier_base_value: float = Field(..., description="Classifier's expected output before considering any features.")
    regressor_base_value: float = Field(..., description="Regressor's expected RUL before considering any features.")
    classifier_waterfall_png_base64: str
    regressor_waterfall_png_base64: str


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
