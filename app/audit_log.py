from __future__ import annotations

import csv
import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

DB_FILENAME = "audit_log.db"
CSV_FILENAME = "audit_log.csv"

_CSV_FIELDS = [
    "timestamp", "asset_id", "input_features", "predicted_class", "predicted_state",
    "confidence", "rul_hours", "model_version", "inference_latency_ms",
]

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    input_features TEXT NOT NULL,
    predicted_class INTEGER,
    predicted_state TEXT,
    confidence REAL,
    rul_hours REAL,
    model_version TEXT,
    inference_latency_ms REAL
)
"""
_CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_predictions_asset_id ON predictions(asset_id)"


@dataclass
class AuditRecord:
    timestamp: str
    asset_id: str
    input_features: dict
    predicted_class: int | None
    predicted_state: str
    confidence: float | None
    rul_hours: float | None
    model_version: str
    inference_latency_ms: float


class AuditLogger:

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.log_dir / DB_FILENAME
        self.csv_path = self.log_dir / CSV_FILENAME
        self._init_db()
        self._init_csv()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(_CREATE_TABLE_SQL)
                conn.execute(_CREATE_INDEX_SQL)
        finally:
            conn.close()

    def _init_csv(self) -> None:
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    def log_prediction(self, record: AuditRecord) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO predictions "
                        "(timestamp, asset_id, input_features, predicted_class, predicted_state, "
                        " confidence, rul_hours, model_version, inference_latency_ms) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            record.timestamp, record.asset_id, json.dumps(record.input_features),
                            record.predicted_class, record.predicted_state, record.confidence,
                            record.rul_hours, record.model_version, record.inference_latency_ms,
                        ),
                    )
            finally:
                conn.close()
        except sqlite3.Error:
            logger.exception("Audit log SQLite write failed for asset_id=%s", record.asset_id)

        try:
            row = asdict(record)
            row["input_features"] = json.dumps(row["input_features"])
            with _LOCK, open(self.csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)
        except OSError:
            logger.exception("Audit log CSV write failed for asset_id=%s", record.asset_id)

    def get_recent(self, limit: int = 100) -> list[dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            finally:
                conn.close()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            logger.exception("Audit log SQLite read (get_recent) failed")
            return []

    def get_by_asset(self, asset_id: str, limit: int = 100) -> list[dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM predictions WHERE asset_id = ? ORDER BY id DESC LIMIT ?",
                    (asset_id, limit),
                ).fetchall()
            finally:
                conn.close()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            logger.exception("Audit log SQLite read (get_by_asset) failed for asset_id=%s", asset_id)
            return []

    def count(self) -> int:
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            finally:
                conn.close()
        except sqlite3.Error:
            logger.exception("Audit log SQLite read (count) failed")
            return 0
