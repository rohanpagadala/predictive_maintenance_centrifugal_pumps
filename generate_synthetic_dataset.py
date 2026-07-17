
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

MACHINE_MODELS = ["Atlas-7", "Hydra-500", "Nova-P", "Titan-X2", "Titan-X3"]
LOCATIONS = [
    "CPF-1",
    "Offshore-Platform-A",
    "Offshore-Platform-B",
    "Onshore-Terminal-1",
    "Onshore-Terminal-2",
]
N_ASSETS = 900

VIB_BASE_MEAN, VIB_BASE_STD = 2.3, 0.35
TEMP_BASE_MEAN, TEMP_BASE_STD = 60.0, 3.0
PRES_BASE_MEAN, PRES_BASE_STD = 145.0, 2.5
FLOW_BASE_MEAN, FLOW_BASE_STD = 250.0, 3.5

VIB_MIN, VIB_MAX = 0.4, 22.0
TEMP_MIN, TEMP_MAX = 35.0, 165.0
PRES_MIN, PRES_MAX = 20.0, 210.0
FLOW_MIN, FLOW_MAX = 0.0, 275.0

WARNING_AT = 0.40
CRITICAL_AT = 0.75

RUL_AT_0 = 9000.0
RUL_AT_WARNING = 2600.0
RUL_AT_CRITICAL = 600.0
RUL_AT_1 = 0.0


def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)




def fault_normal(s: np.ndarray, rng: np.random.Generator):
    n = len(s)
    return (
        np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
        np.ones(n), np.ones(n), np.ones(n),
    )


def fault_bearing(s, rng):
    n = len(s)
    vib = 8.0 * s ** 1.1
    temp = 24.0 * s
    pres = np.zeros(n)
    flow = -5.0 * s ** 1.5
    return vib, temp, pres, flow, np.ones(n), np.ones(n), np.ones(n)


def fault_seal(s, rng):
    n = len(s)
    vib = 1.5 * s
    temp = 6.0 * s
    pres = -28.0 * s
    flow = -38.0 * s
    return vib, temp, pres, flow, np.ones(n), np.ones(n), np.ones(n)


def fault_motor(s, rng):
    n = len(s)
    vib = 2.0 * s
    temp = 48.0 * s ** 1.05
    pres = -28.0 * s
    flow = -58.0 * s
    return vib, temp, pres, flow, np.ones(n), np.ones(n), np.ones(n)


def fault_cavitation(s, rng):
    n = len(s)
    vib = 3.5 * s
    temp = 5.0 * s
    pres = -20.0 * s
    flow = -48.0 * s
    vib_noise = 1.0 + 4.0 * s
    pres_noise = 1.0 + 5.0 * s
    return vib, temp, pres, flow, vib_noise, pres_noise, np.ones(n)


def fault_impeller(s, rng):
    n = len(s)
    vib = 4.0 * s
    temp = 2.0 * s
    pres = -24.0 * s
    flow = -42.0 * s
    return vib, temp, pres, flow, np.ones(n), np.ones(n), np.ones(n)


def fault_misalignment(s, rng):
    n = len(s)
    vib = 7.5 * s
    temp = 16.0 * s
    pres = np.zeros(n)
    flow = -6.0 * s
    return vib, temp, pres, flow, np.ones(n), np.ones(n), np.ones(n)


def fault_lubrication(s, rng):
    n = len(s)
    vib = 4.5 * s
    temp = 20.0 * s
    pres = -2.0 * s
    flow = -3.0 * s
    return vib, temp, pres, flow, np.ones(n), np.ones(n), np.ones(n)


def fault_rotor_imbalance(s, rng):
    n = len(s)
    vib = 8.5 * s
    temp = 3.0 * s
    pres = np.zeros(n)
    flow = np.zeros(n)
    return vib, temp, pres, flow, np.ones(n), np.ones(n), np.ones(n)


def fault_coupling(s, rng):
    n = len(s)
    vib = 6.5 * s
    temp = 3.0 * s
    pres = -14.0 * s
    flow = -18.0 * s
    vib_noise = 1.0 + 2.0 * s
    return vib, temp, pres, flow, vib_noise, np.ones(n), 1.0 + 1.5 * s


def fault_suction_blockage(s, rng):
    n = len(s)
    vib = 3.0 * s
    temp = 3.0 * s
    pres = -32.0 * s
    flow = -55.0 * s
    vib_noise = 1.0 + 1.5 * s
    return vib, temp, pres, flow, vib_noise, np.ones(n), np.ones(n)


def fault_discharge_blockage(s, rng):
    n = len(s)
    vib = 2.0 * s
    temp = 13.0 * s
    pres = 32.0 * s
    flow = -50.0 * s
    return vib, temp, pres, flow, np.ones(n), np.ones(n), np.ones(n)


def fault_valve_malfunction(s, rng):
    n = len(s)
    vib = 3.0 * s
    temp = np.zeros(n)
    pres = rng.normal(0, 1, n) * 6.0 * s
    flow = rng.normal(0, 1, n) * 8.0 * s
    pres_noise = 1.0 + 3.0 * s
    flow_noise = 1.0 + 3.0 * s
    return vib, temp, pres, flow, np.ones(n), pres_noise, flow_noise


def fault_overheating(s, rng):
    n = len(s)
    vib = 1.5 * s
    temp = 58.0 * s ** 1.05
    pres = -4.0 * s
    flow = -9.0 * s
    return vib, temp, pres, flow, np.ones(n), np.ones(n), np.ones(n)


def fault_corrosion_wear(s, rng):
    n = len(s)
    vib = 4.0 * s
    temp = 4.0 * s
    pres = -18.0 * s
    flow = -18.0 * s
    return vib, temp, pres, flow, np.ones(n), np.ones(n), np.ones(n)


FAULT_MODELS = {
    "Normal": (fault_normal, (1.5, 6.0), 0.35),
    "Bearing Failure": (fault_bearing, (2.0, 2.0), 1.0),
    "Seal Failure": (fault_seal, (2.0, 2.0), 1.0),
    "Motor Failure": (fault_motor, (2.0, 2.0), 1.0),
    "Cavitation": (fault_cavitation, (2.0, 2.0), 1.0),
    "Impeller Damage": (fault_impeller, (2.0, 2.0), 1.0),
    "Shaft Misalignment": (fault_misalignment, (2.0, 2.0), 1.0),
    "Lubrication Failure": (fault_lubrication, (2.0, 2.0), 1.0),
    "Rotor Imbalance": (fault_rotor_imbalance, (2.0, 2.0), 1.0),
    "Coupling Failure": (fault_coupling, (2.0, 2.0), 1.0),
    "Suction Blockage": (fault_suction_blockage, (2.0, 2.0), 1.0),
    "Discharge Blockage": (fault_discharge_blockage, (2.0, 2.0), 1.0),
    "Valve Malfunction": (fault_valve_malfunction, (2.0, 2.0), 1.0),
    "Overheating": (fault_overheating, (2.0, 2.0), 1.0),
    "Corrosion / Wear": (fault_corrosion_wear, (2.0, 2.0), 1.0),
}

FAULT_PROGRESSION_HOURS = {
    "Normal": (0, 0),
    "Bearing Failure": (500, 4000),
    "Seal Failure": (300, 2500),
    "Motor Failure": (100, 1500),
    "Cavitation": (50, 800),
    "Impeller Damage": (300, 3000),
    "Shaft Misalignment": (200, 2000),
    "Lubrication Failure": (400, 3500),
    "Rotor Imbalance": (200, 2000),
    "Coupling Failure": (20, 500),
    "Suction Blockage": (20, 600),
    "Discharge Blockage": (50, 900),
    "Valve Malfunction": (50, 1200),
    "Overheating": (50, 1000),
    "Corrosion / Wear": (1000, 8000),
}


def build_asset_registry(rng: np.random.Generator) -> pd.DataFrame:
    asset_ids = [f"PUMP_{i:04d}" for i in range(1, N_ASSETS + 1)]
    return pd.DataFrame(
        {
            "Asset_ID": asset_ids,
            "Machine_Model": rng.choice(MACHINE_MODELS, size=N_ASSETS),
            "Location": rng.choice(LOCATIONS, size=N_ASSETS),
        }
    )


def rul_from_severity(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    breakpoints = [0.0, WARNING_AT, CRITICAL_AT, 1.0]
    values = [RUL_AT_0, RUL_AT_WARNING, RUL_AT_CRITICAL, RUL_AT_1]
    mean_rul = np.interp(s, breakpoints, values)
    noisy = mean_rul * rng.normal(1.0, 0.08, size=len(s))
    noisy = np.clip(noisy, 0, None)

    normal_mask = s < WARNING_AT
    warning_mask = (s >= WARNING_AT) & (s < CRITICAL_AT)
    critical_mask = s >= CRITICAL_AT

    out = noisy.copy()
    out[normal_mask] = _clip(out[normal_mask], RUL_AT_WARNING, RUL_AT_0)
    out[warning_mask] = _clip(out[warning_mask], RUL_AT_CRITICAL, RUL_AT_WARNING)
    out[critical_mask] = _clip(out[critical_mask], RUL_AT_1, RUL_AT_CRITICAL)
    return out


def health_status_from_severity(s: np.ndarray) -> np.ndarray:
    status = np.full(len(s), "Normal", dtype=object)
    status[(s >= WARNING_AT) & (s < CRITICAL_AT)] = "Warning"
    status[s >= CRITICAL_AT] = "Critical"
    return status


def generate_fault_block(
    fault_name: str,
    n: int,
    asset_registry: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    model_fn, (alpha, beta), sev_cap = FAULT_MODELS[fault_name]

    severity = rng.beta(alpha, beta, size=n) * sev_cap

    assets = asset_registry.sample(n=n, replace=True, random_state=rng.integers(0, 2**31 - 1)).reset_index(drop=True)

    vib_base = _clip(rng.normal(VIB_BASE_MEAN, VIB_BASE_STD, n), 1.2, 3.4)
    temp_base = _clip(rng.normal(TEMP_BASE_MEAN, TEMP_BASE_STD, n), 52, 68)
    pres_base = _clip(rng.normal(PRES_BASE_MEAN, PRES_BASE_STD, n), 135, 150)
    flow_base = _clip(rng.normal(FLOW_BASE_MEAN, FLOW_BASE_STD, n), 235, 258)

    vib_d, temp_d, pres_d, flow_d, vib_nm, pres_nm, flow_nm = model_fn(severity, rng)

    vibration = vib_base + vib_d + rng.normal(0, 0.15, n) * vib_nm
    temperature = temp_base + temp_d + rng.normal(0, 0.8, n)
    pressure = pres_base + pres_d + rng.normal(0, 1.2, n) * pres_nm
    flow_rate = flow_base + flow_d + rng.normal(0, 1.5, n) * flow_nm

    vibration = _clip(vibration, VIB_MIN, VIB_MAX)
    temperature = _clip(temperature, TEMP_MIN, TEMP_MAX)
    pressure = _clip(pressure, PRES_MIN, PRES_MAX)
    flow_rate = _clip(flow_rate, FLOW_MIN, FLOW_MAX)

    lo, hi = FAULT_PROGRESSION_HOURS[fault_name]
    progression_hours = rng.uniform(lo, hi, n) * severity if hi > 0 else np.zeros(n)
    install_hours = rng.uniform(300, 30000, n)
    operating_hours = _clip(install_hours + progression_hours, 0, 60000)

    rul = rul_from_severity(severity, rng)
    health = health_status_from_severity(severity)

    return pd.DataFrame(
        {
            "Asset_ID": assets["Asset_ID"],
            "Machine_Model": assets["Machine_Model"],
            "Location": assets["Location"],
            "Vibration": np.round(vibration, 2),
            "Temperature": np.round(temperature, 1),
            "Pressure": np.round(pressure, 1),
            "Flow_Rate": np.round(flow_rate, 1),
            "Operating_Hours": np.round(operating_hours, 0).astype(int),
            "Health_Status": health,
            "Remaining_Useful_Life_Hours": np.round(rul, 0).astype(int),
            "Fault_Type": fault_name,
        }
    )


def generate_dataset(total_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    asset_registry = build_asset_registry(rng)

    fault_names = list(FAULT_MODELS.keys())
    n_faults = len(fault_names) - 1

    normal_share = 0.22
    n_normal = int(round(total_rows * normal_share))
    n_per_fault = (total_rows - n_normal) // n_faults
    n_normal = total_rows - n_per_fault * n_faults

    blocks = [generate_fault_block("Normal", n_normal, asset_registry, rng)]
    for name in fault_names:
        if name == "Normal":
            continue
        blocks.append(generate_fault_block(name, n_per_fault, asset_registry, rng))

    df = pd.concat(blocks, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=36000, help="Total number of synthetic records (20000-50000).")
    parser.add_argument(
        "--out",
        type=str,
        default=str(BASE_DIR / "data" / "synthetic_pump_dataset.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (20_000 <= args.rows <= 50_000):
        raise ValueError("--rows must be between 20000 and 50000")

    df = generate_dataset(args.rows, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df):,} rows to {out_path}")
    print("\nFault_Type distribution:")
    print(df["Fault_Type"].value_counts())
    print("\nHealth_Status distribution:")
    print(df["Health_Status"].value_counts())
    print("\nRUL by Health_Status (min/mean/max):")
    print(df.groupby("Health_Status")["Remaining_Useful_Life_Hours"].agg(["min", "mean", "max"]))


if __name__ == "__main__":
    main()
