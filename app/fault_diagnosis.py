from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FAULT_BEARING = "Bearing_Failure"
FAULT_MOTOR = "Motor_Failure"
FAULT_SEAL = "Seal_Failure"
FAULT_UNKNOWN = "Unknown_Critical_Fault"
SPECIFIC_FAULTS = {FAULT_BEARING, FAULT_MOTOR, FAULT_SEAL}

DOMINANCE_MARGIN = 1.15

_READABLE_CHANNEL = {
    "Vibration_mm_s": "vibration",
    "Temperature_C": "temperature",
    "Pressure_psi": "pressure",
    "Flow_Rate_m3_h": "flow rate",
}


@dataclass
class FaultDiagnosisResult:
    predicted_fault: str
    confidence: float
    dominant_channel: str
    channel_deviation_scores: dict[str, float]
    rationale: str


def _channel_z_scores(row: pd.Series, baseline: dict[str, dict[str, float]]) -> dict[str, float]:
    scores = {}
    for col, stats in baseline.items():
        std = stats["std"] or 1e-9
        scores[col] = abs((float(row[col]) - stats["mean"]) / std)
    return scores


def diagnose(row: pd.Series, baseline: dict[str, dict[str, float]]) -> FaultDiagnosisResult:
    z = _channel_z_scores(row, baseline)

    fault_signals = {
        FAULT_BEARING: (z.get("Vibration_mm_s", 0.0), "Vibration_mm_s"),
        FAULT_MOTOR: (z.get("Temperature_C", 0.0), "Temperature_C"),
        FAULT_SEAL: ((z.get("Pressure_psi", 0.0) + z.get("Flow_Rate_m3_h", 0.0)) / 2, "Pressure_psi & Flow_Rate_m3_h"),
    }
    ranked = sorted(fault_signals.items(), key=lambda kv: kv[1][0], reverse=True)
    (top_fault, (top_score, top_channel)), (_, (runner_up_score, _)) = ranked[0], ranked[1]

    if top_score <= 0 or top_score < DOMINANCE_MARGIN * runner_up_score:
        return FaultDiagnosisResult(
            predicted_fault=FAULT_UNKNOWN,
            confidence=0.0 if top_score <= 0 else round(1.0 - runner_up_score / top_score, 4),
            dominant_channel="none (ambiguous)",
            channel_deviation_scores={k: round(v, 4) for k, v in z.items()},
            rationale=(
                "No single sensor channel's deviation from the Normal-state baseline "
                "clearly dominates the others -- this Critical reading's pattern doesn't "
                "resemble one specific historical failure mode strongly enough to name it."
            ),
        )

    return FaultDiagnosisResult(
        predicted_fault=top_fault,
        confidence=round(1.0 - runner_up_score / top_score, 4),
        dominant_channel=top_channel,
        channel_deviation_scores={k: round(v, 4) for k, v in z.items()},
        rationale=(
            f"{top_fault.replace('_', ' ')}-like pattern: deviation from the Normal-state "
            f"baseline is dominated by {_READABLE_CHANNEL.get(top_channel, top_channel)} "
            f"({top_score:.1f} std devs vs. {runner_up_score:.1f} for the next channel)."
        ),
    )
