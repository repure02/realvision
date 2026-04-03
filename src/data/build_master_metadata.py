from pathlib import Path
import pandas as pd

from src.utils.config import get_paths_config, load_dataset_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG = load_dataset_config()
PATHS_CONFIG = get_paths_config(CONFIG)

METADATA_DIR = PATHS_CONFIG.get("metadata_dir") or (PROJECT_ROOT / "data" / "metadata")
OUTPUT_PATH = PATHS_CONFIG.get("master_metadata_csv") or (METADATA_DIR / "master_metadata.csv")

INPUT_FILES = [
    METADATA_DIR / "pexels_metadata.csv",
    METADATA_DIR / "wikimedia_metadata.csv",
    METADATA_DIR / "defactify_metadata.csv",
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    df = pd.read_csv(path)
    df["metadata_file"] = path.name
    return df


def add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["filepath"] = df["filepath"].astype(str)
    df["filename"] = df["filepath"].apply(lambda x: Path(x).name)

    df["is_ai"] = df["label"].map({
        "real": 0,
        "ai_generated": 1,
    })

    df["width"] = pd.to_numeric(df["width"], errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")

    df["aspect_ratio"] = df["width"] / df["height"]
    df["megapixels"] = (df["width"] * df["height"]) / 1_000_000

    df["exists_on_disk"] = df["filepath"].apply(
        lambda x: (PROJECT_ROOT / x).exists()
    )

    df["min_side"] = df[["width", "height"]].min(axis=1)
    df["max_side"] = df[["width", "height"]].max(axis=1)
    df["is_square"] = df["width"] == df["height"]

    return df


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = {
        "collection_date",
        "pexels_url",
        "photographer",
        "query",
        "metadata_file",
        "page_title",
        "description_url",
        "uploader",
        "license_short_name",
        "caption",
        "label_a",
        "label_b",
        "file_ext_from_path",
        "processed_filepath",
    }
    drop_list = [col for col in df.columns if col.lower() in to_drop]
    if drop_list:
        return df.drop(columns=drop_list)
    return df


def run_sanity_checks(df: pd.DataFrame) -> None:
    print("\n=== SANITY CHECKS ===")

    print(f"Total rows: {len(df)}")
    print("\nLabel counts:")
    print(df["label"].value_counts(dropna=False))

    print("\nSource counts:")
    print(df["source"].value_counts(dropna=False))

    if "generator_name" in df.columns:
        print("\nTop generator_name counts:")
        print(df["generator_name"].fillna("MISSING").value_counts(dropna=False).head(20))

    missing_files = (~df["exists_on_disk"]).sum()
    print(f"\nMissing files on disk: {missing_files}")

    duplicate_image_ids = df["image_id"].duplicated().sum()
    duplicate_filepaths = df["filepath"].duplicated().sum()
    print(f"Duplicate image_id rows: {duplicate_image_ids}")
    print(f"Duplicate filepath rows: {duplicate_filepaths}")

    invalid_labels = df["is_ai"].isna().sum()
    print(f"Rows with unmapped labels: {invalid_labels}")

    print("\nFormat counts:")
    if "format" in df.columns:
        print(df["format"].value_counts(dropna=False))

    print("\nRows with missing width/height:")
    print(df[["width", "height"]].isna().any(axis=1).sum())


def main() -> None:
    dfs = [load_csv(path) for path in INPUT_FILES]
    master_df = pd.concat(dfs, ignore_index=True)

    master_df = add_helper_columns(master_df)
    master_df = drop_unwanted_columns(master_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved master metadata to: {OUTPUT_PATH}")
    run_sanity_checks(master_df)


if __name__ == "__main__":
    main()
