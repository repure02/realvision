from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import get_paths_config, load_dataset_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG = load_dataset_config()
PATHS_CONFIG = get_paths_config(CONFIG)

METADATA_DIR = PATHS_CONFIG.get("metadata_dir") or (PROJECT_ROOT / "data" / "metadata")
INPUT_METADATA = PATHS_CONFIG.get("processed_metadata_csv") or (METADATA_DIR / "processed_metadata.csv")
OUTPUT_METADATA = INPUT_METADATA

RANDOM_SEED = 42


def create_random_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    df["random_split"] = None
    df.loc[train_df.index, "random_split"] = "train"
    df.loc[val_df.index, "random_split"] = "val"
    df.loc[test_df.index, "random_split"] = "test"

    return df


def print_split_summary(df: pd.DataFrame):
    print("\n=== RANDOM SPLIT SUMMARY ===")
    print(pd.crosstab(df["random_split"], df["label"]))


def main():
    parser = argparse.ArgumentParser(
        description="Create the random split metadata used as the real-image partition for LOGO."
    )
    parser.parse_args()

    df = pd.read_csv(INPUT_METADATA)
    df = create_random_split(df)
    df.to_csv(OUTPUT_METADATA, index=False)

    print(f"\nSaved metadata with splits to: {OUTPUT_METADATA}")
    print_split_summary(df)


if __name__ == "__main__":
    main()
