from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import DEFAULT_DATASET_CONFIG, load_dataset_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
REGISTRY_JSONL = RUNS_DIR / "registry.jsonl"
REGISTRY_CSV = RUNS_DIR / "registry.csv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_").lower() or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{slug}"


def _to_repo_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return _to_repo_relative(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def args_to_dict(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if hasattr(args, "__dict__"):
        return {k: _sanitize(v) for k, v in vars(args).items()}
    return {}


def get_dataset_context(metadata_path: Path | None) -> dict[str, Any]:
    config = load_dataset_config()
    metadata_rows = None
    metadata_columns = None
    metadata_mtime_utc = None
    if metadata_path and metadata_path.exists():
        df = pd.read_csv(metadata_path)
        metadata_rows = int(len(df))
        metadata_columns = sorted(df.columns.astype(str).tolist())
        metadata_mtime_utc = (
            datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )

    return {
        "dataset_name": str(config.get("dataset_name", "unknown")),
        "dataset_version": str(config.get("dataset_name", "unknown")),
        "dataset_config_path": _to_repo_relative(DEFAULT_DATASET_CONFIG),
        "dataset_config": _sanitize(config),
        "metadata_path": _to_repo_relative(metadata_path) if metadata_path else None,
        "metadata_rows": metadata_rows,
        "metadata_columns": metadata_columns,
        "metadata_mtime_utc": metadata_mtime_utc,
    }


def build_run_record(
    *,
    run_name: str,
    task: str,
    split_type: str,
    args: Any,
    metadata_path: Path | None,
    checkpoint_path: Path | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    status: str = "completed",
    extra: dict[str, Any] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    run_id = make_run_id(run_name)
    started_at = started_at or utc_now_iso()
    completed_at = completed_at or utc_now_iso()
    checkpoint_name = checkpoint_path.name if checkpoint_path else None

    record = {
        "run_id": run_id,
        "run_name": run_name,
        "task": task,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "timestamp": completed_at,
        "split_type": split_type,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": _to_repo_relative(checkpoint_path),
        "metrics": _sanitize(metrics or {}),
        "artifacts": _sanitize(artifacts or {}),
        "args": args_to_dict(args),
    }
    record.update(get_dataset_context(metadata_path))
    if extra:
        record.update(_sanitize(extra))
    return record


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "run_id": record.get("run_id"),
        "run_name": record.get("run_name"),
        "task": record.get("task"),
        "status": record.get("status"),
        "timestamp": record.get("timestamp"),
        "started_at": record.get("started_at"),
        "completed_at": record.get("completed_at"),
        "dataset_version": record.get("dataset_version"),
        "dataset_name": record.get("dataset_name"),
        "metadata_path": record.get("metadata_path"),
        "metadata_rows": record.get("metadata_rows"),
        "split_type": record.get("split_type"),
        "checkpoint_name": record.get("checkpoint_name"),
        "checkpoint_path": record.get("checkpoint_path"),
    }

    metrics = record.get("metrics", {})
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            flat[f"metric_{key}"] = value

    args = record.get("args", {})
    if isinstance(args, dict):
        for key in (
            "epochs",
            "loss",
            "focal_gamma",
            "final_inference",
            "final_val_fraction",
            "split_column",
            "run_tag",
            "threshold_sweep",
            "target_recall",
            "decision_threshold",
            "checkpoint_name",
            "checkpoint_path",
            "logo",
            "logo_all",
            "logo_test_generator",
            "logo_val_generator",
        ):
            if key in args:
                flat[f"arg_{key}"] = args[key]

    for key in (
        "val_generator",
        "test_generator",
        "run_group",
        "model_name",
        "device",
        "registry_file",
    ):
        if key in record:
            flat[key] = record[key]

    return flat


def log_run(record: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_path = RUNS_DIR / f"{record['run_id']}.json"
    run_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    with REGISTRY_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    records = []
    with REGISTRY_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    flat_rows = [_flatten_record(item) for item in records]
    if flat_rows:
        fieldnames = sorted({key for row in flat_rows for key in row.keys()})
        with REGISTRY_CSV.open("w", encoding="utf-8", newline="") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_rows:
                writer.writerow(row)

    return run_path
