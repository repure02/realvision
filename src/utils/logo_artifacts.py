from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
DETAILS_DIR = REPORTS_DIR / "details"


def get_logo_details_dir() -> Path:
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    return DETAILS_DIR


def logo_generator_metrics_path(generator_name: str) -> Path:
    return get_logo_details_dir() / f"logo_test_{generator_name}_generator_metrics.csv"


def logo_predictions_path(generator_name: str) -> Path:
    return get_logo_details_dir() / f"logo_test_{generator_name}_predictions.csv"


def to_repo_relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
