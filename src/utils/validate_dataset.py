from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import get_training_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = {
    "image_id",
    "label",
    "source",
    "generator_name",
    "random_split",
}
VALID_LABELS = {"real", "ai_generated"}
VALID_SPLITS = {"train", "val", "test"}


def resolve_metadata_path(path_arg: str | None) -> Path | None:
    if path_arg:
        path = Path(path_arg)
        return path if path.is_absolute() else PROJECT_ROOT / path
    _, _, _, metadata_path = get_training_settings()
    return metadata_path


def validate_dataset(metadata_path: Path, check_files: bool = False) -> None:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Metadata is empty.")

    labels = set(df["label"].dropna().astype(str).unique())
    invalid_labels = labels - VALID_LABELS
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {sorted(invalid_labels)}")

    splits = set(df["random_split"].dropna().astype(str).unique())
    invalid_splits = splits - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"Invalid random_split values found: {sorted(invalid_splits)}")

    if df["image_id"].astype(str).duplicated().any():
        duplicates = sorted(
            df.loc[df["image_id"].astype(str).duplicated(), "image_id"].astype(str).unique()
        )
        raise ValueError(f"Duplicate image_id values found: {duplicates[:10]}")

    ai_df = df[df["label"] == "ai_generated"]
    if ai_df["generator_name"].isna().any() or (ai_df["generator_name"].astype(str) == "").any():
        raise ValueError("AI-generated rows must include generator_name.")

    real_generators = set(df.loc[df["label"] == "real", "generator_name"].fillna("none").astype(str))
    if real_generators - {"none"}:
        raise ValueError(f"Real rows should use generator_name='none': {sorted(real_generators)}")

    filepath_column = "processed_filepath" if "processed_filepath" in df.columns else "filepath"
    if filepath_column not in df.columns:
        raise ValueError("Metadata must include either 'processed_filepath' or 'filepath'.")

    if check_files:
        missing_paths = []
        for value in df[filepath_column].dropna().astype(str):
            if not (PROJECT_ROOT / value).exists():
                missing_paths.append(value)
                if len(missing_paths) >= 10:
                    break
        if missing_paths:
            raise FileNotFoundError(
                "Missing image files referenced by metadata, first examples: "
                + ", ".join(missing_paths)
            )

    label_counts = df["label"].value_counts().to_dict()
    split_counts = df["random_split"].value_counts().to_dict()
    generator_counts = ai_df["generator_name"].value_counts().to_dict()

    print(f"Validated dataset metadata: {metadata_path.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(df)}")
    print(f"Labels: {label_counts}")
    print(f"Random splits: {split_counts}")
    print(f"AI generators: {generator_counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RealVision dataset metadata.")
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument(
        "--check_files",
        action="store_true",
        help="Also verify that every referenced processed image exists on disk.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Exit successfully if metadata is not present, useful for lightweight CI clones.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = resolve_metadata_path(args.metadata_path)
    if metadata_path is None or not metadata_path.exists():
        if args.allow_missing:
            print("Dataset metadata not found; skipping dataset validation.")
            return
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    validate_dataset(metadata_path, check_files=args.check_files)


if __name__ == "__main__":
    main()
