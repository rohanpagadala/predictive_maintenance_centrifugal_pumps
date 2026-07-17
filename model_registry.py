from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import model_persistence as mp

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "registry.json"


class ModelRegistryError(RuntimeError):
    pass


def _get_git_commit_hash(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=repo_dir, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _load_registry(models_root: Path) -> dict:
    path = models_root / REGISTRY_FILENAME
    if not path.exists():
        raise ModelRegistryError(
            f"No registry at {path}. Call register_version() or migrate_flat_models_to_v1() first."
        )
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ModelRegistryError(f"Registry at {path} is corrupted: {exc}") from exc


def _save_registry(models_root: Path, registry: dict) -> None:
    path = models_root / REGISTRY_FILENAME
    mp.atomic_write_text(path, json.dumps(registry, indent=2))


def _next_version(registry: dict) -> str:
    existing = [int(v[1:]) for v in registry["versions"] if v.startswith("v") and v[1:].isdigit()]
    return f"v{(max(existing) + 1) if existing else 1}"


def register_version(
    classifier: Any, classifier_name: str,
    regressor: Any, regressor_name: str,
    scaler: Any, encoders: Any,
    feature_list: list[str], class_names: list[str],
    feature_engineering_config: dict,
    models_root: str | Path,
    dataset_version: str,
    performance_metrics: dict,
    repo_dir: str | Path,
    version: str | None = None,
    promote_to_stable: bool = False,
    fault_diagnosis_baseline: dict | None = None,
) -> str:
    models_root = Path(models_root)
    models_root.mkdir(parents=True, exist_ok=True)
    registry_path = models_root / REGISTRY_FILENAME
    if not registry_path.exists():
        registry = {"latest": None, "stable": None, "stable_history": [], "versions": {}}
    else:
        try:
            registry = json.loads(registry_path.read_text())
        except json.JSONDecodeError as exc:
            raise ModelRegistryError(f"Registry at {registry_path} is corrupted: {exc}") from exc

    version = version or _next_version(registry)
    version_dir = models_root / version

    extra_metadata = {
        "model_version": version,
        "training_date": performance_metrics.pop("training_date", None)
        or datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "git_commit_hash": _get_git_commit_hash(Path(repo_dir)),
        **performance_metrics,
    }

    mp.save_models(
        classifier=classifier, classifier_name=classifier_name,
        regressor=regressor, regressor_name=regressor_name,
        scaler=scaler, encoders=encoders,
        feature_list=feature_list, class_names=class_names,
        feature_engineering_config=feature_engineering_config,
        models_dir=version_dir, extra_metadata=extra_metadata,
        fault_diagnosis_baseline=fault_diagnosis_baseline,
    )

    registry["versions"][version] = extra_metadata
    registry["latest"] = version
    if promote_to_stable or registry["stable"] is None:
        registry["stable"] = version
        registry.setdefault("stable_history", []).append(version)
    _save_registry(models_root, registry)
    logger.info("Registered model version %s (stable=%s, latest=%s)", version, registry["stable"], registry["latest"])
    return version


def migrate_flat_models_to_v1(flat_dir: str | Path, models_root: str | Path, repo_dir: str | Path) -> str:
    flat_dir = Path(flat_dir)
    artifacts = mp.load_models(flat_dir)
    old_metadata = artifacts.metadata

    performance_metrics = {
        k: v for k, v in old_metadata.items()
        if k not in {"classifier_name", "classifier_filename", "regressor_name",
                      "regressor_filename", "n_features", "class_names", "saved_at_utc", "joblib_version"}
    }
    performance_metrics["training_date"] = old_metadata.get("saved_at_utc")

    return register_version(
        classifier=artifacts.classifier, classifier_name=artifacts.classifier_name,
        regressor=artifacts.regressor, regressor_name=artifacts.regressor_name,
        scaler=artifacts.scaler, encoders=artifacts.encoders,
        feature_list=artifacts.feature_list, class_names=artifacts.class_names,
        feature_engineering_config=artifacts.feature_engineering_config,
        models_root=models_root,
        dataset_version="AI_Predictive_Maintenance_Pump_Dataset_10000.xlsx",
        performance_metrics=performance_metrics,
        repo_dir=repo_dir,
        version="v1",
        promote_to_stable=True,
        fault_diagnosis_baseline=artifacts.fault_diagnosis_baseline,
    )


def get_active_version(models_root: str | Path) -> str:
    registry = _load_registry(Path(models_root))
    if not registry["stable"]:
        raise ModelRegistryError("No stable version set.")
    return registry["stable"]


def get_version_dir(models_root: str | Path, version: str) -> Path:
    models_root = Path(models_root)
    registry = _load_registry(models_root)
    if version not in registry["versions"]:
        raise ModelRegistryError(f"Unknown model version {version!r}. Known: {sorted(registry['versions'])}")
    return models_root / version


def _verify_version_artifacts(models_root: Path, version: str) -> None:
    version_dir = models_root / version
    metadata_path = version_dir / mp.METADATA_FILENAME
    if not metadata_path.exists():
        raise ModelRegistryError(f"Cannot promote/roll back to {version!r}: {metadata_path} is missing.")
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise ModelRegistryError(f"Cannot promote/roll back to {version!r}: {metadata_path} is corrupted: {exc}") from exc

    required_files = [
        metadata.get("classifier_filename"), metadata.get("regressor_filename"),
        mp.SCALER_FILENAME, mp.ENCODERS_FILENAME, mp.FEATURE_LIST_FILENAME,
        mp.CLASS_NAMES_FILENAME, mp.FEATURE_ENGINEERING_CONFIG_FILENAME,
    ]
    missing = [f for f in required_files if not f or not (version_dir / f).exists()]
    if missing:
        raise ModelRegistryError(
            f"Cannot promote/roll back to {version!r}: missing artifact file(s) in {version_dir}: {missing}"
        )


def list_versions(models_root: str | Path) -> list[dict]:
    registry = _load_registry(Path(models_root))
    return [
        {"version": v, "is_stable": v == registry["stable"], "is_latest": v == registry["latest"], **meta}
        for v, meta in sorted(registry["versions"].items())
    ]


def promote_to_stable(models_root: str | Path, version: str) -> None:
    models_root = Path(models_root)
    registry = _load_registry(models_root)
    if version not in registry["versions"]:
        raise ModelRegistryError(f"Unknown model version {version!r}. Known: {sorted(registry['versions'])}")
    _verify_version_artifacts(models_root, version)
    registry["stable"] = version
    registry.setdefault("stable_history", []).append(version)
    _save_registry(models_root, registry)
    logger.info("Promoted %s to stable.", version)


def rollback_to_previous(models_root: str | Path) -> str:
    models_root = Path(models_root)
    registry = _load_registry(models_root)
    history = registry.get("stable_history", [])
    if len(history) < 2:
        raise ModelRegistryError("No previous stable version to roll back to.")
    history.pop()
    previous = history[-1]
    _verify_version_artifacts(models_root, previous)
    registry["stable"] = previous
    registry["stable_history"] = history
    _save_registry(models_root, registry)
    logger.warning("Rolled back stable model to %s.", previous)
    return previous


def load_models_from_registry(models_root: str | Path, version: str | None = None) -> mp.ModelArtifacts:
    models_root = Path(models_root)
    resolved = version or get_active_version(models_root)
    version_dir = get_version_dir(models_root, resolved)
    artifacts = mp.load_models(version_dir)
    logger.info("Loaded model version %s from %s", resolved, version_dir)
    return artifacts
