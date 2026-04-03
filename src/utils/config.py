from pathlib import Path
import os
from typing import Any, Dict, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "data" / "dataset_v1.yaml"


def load_dataset_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_DATASET_CONFIG
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_env_or_config(
    key: str,
    config: Dict[str, Any],
    default: Optional[str] = None,
) -> Optional[str]:
    if key in os.environ:
        return os.environ[key]
    env_config = config.get("env", {}) if config else {}
    value = env_config.get(key)
    if value is None or value == "":
        return default
    return str(value)


def get_training_settings(config: Optional[Dict[str, Any]] = None):
    config = config or load_dataset_config()

    image_size_default = str(config.get("image_size", 224))

    batch_size = int(get_env_or_config("REALVISION_BATCH_SIZE", config, "32"))
    num_workers = int(get_env_or_config("REALVISION_NUM_WORKERS", config, "0"))
    image_size = int(get_env_or_config("REALVISION_IMAGE_SIZE", config, image_size_default))

    metadata_path_value = get_env_or_config("REALVISION_METADATA_PATH", config, "")
    if not metadata_path_value:
        paths = config.get("paths", {})
        metadata_path_value = (
            paths.get("processed_metadata_csv")
            or paths.get("master_metadata_csv")
            or ""
        )

    metadata_path = _resolve_path(metadata_path_value) if metadata_path_value else None
    if metadata_path and metadata_path.name == "master_metadata.csv":
        candidate = metadata_path.parent / "processed_metadata.csv"
        if candidate.exists():
            metadata_path = candidate

    return batch_size, num_workers, image_size, metadata_path


def get_paths_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[Path]]:
    config = config or load_dataset_config()
    paths = config.get("paths", {})

    def resolve_optional(key: str) -> Optional[Path]:
        value = paths.get(key)
        if not value:
            return None
        return _resolve_path(str(value))

    return {
        "raw_real_dir": resolve_optional("raw_real_dir"),
        "raw_ai_dir": resolve_optional("raw_ai_dir"),
        "metadata_dir": resolve_optional("metadata_dir"),
        "master_metadata_csv": resolve_optional("master_metadata_csv"),
        "processed_metadata_csv": resolve_optional("processed_metadata_csv"),
        "processed_images_dir": resolve_optional("processed_images_dir"),
    }


def get_processing_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    config = config or load_dataset_config()
    processing = config.get("processing", {})
    return {
        "max_side": int(processing.get("max_side", 512)),
        "jpeg_quality": int(processing.get("jpeg_quality", 90)),
    }
