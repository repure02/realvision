from pathlib import Path

import pandas as pd

from src.utils.config import get_paths_config, load_dataset_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

CONFIG = load_dataset_config()
PATHS_CONFIG = get_paths_config(CONFIG)
METADATA_PATH = (
    PATHS_CONFIG.get("processed_metadata_csv")
    or PATHS_CONFIG.get("master_metadata_csv")
    or (PROJECT_ROOT / "data" / "metadata" / "processed_metadata.csv")
)


def main() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)
    if "label" not in df.columns:
        raise ValueError("Metadata must include a 'label' column.")

    ai_df = df[df["label"] == "ai_generated"].copy()
    if "generator_family" not in ai_df.columns:
        raise ValueError("Metadata must include a 'generator_family' column.")

    ai_df["generator_family"] = (
        ai_df["generator_family"]
        .fillna("unknown")
        .replace("", "unknown")
        .astype(str)
    )
    if "generator_name" not in ai_df.columns:
        raise ValueError("Metadata must include a 'generator_name' column.")
    ai_df["generator_name"] = (
        ai_df["generator_name"]
        .fillna("unknown")
        .replace("", "unknown")
        .astype(str)
    )

    summary_df = (
        ai_df.groupby("generator_name", dropna=False)
        .size()
        .reset_index(name="image_count")
        .sort_values(["image_count", "generator_name"], ascending=[False, True])
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "dataset_specs.csv"
    summary_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
