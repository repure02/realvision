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


def create_heldout_generator_split(
    df: pd.DataFrame,
    heldout_test_generator: str | None = None,
    heldout_val_generator: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["heldout_generator_split"] = None

    # Real images are always train/val/test split randomly
    real_df = df[df["label"] == "real"]

    real_train, real_temp = train_test_split(
        real_df,
        test_size=0.30,
        stratify=real_df["label"],
        random_state=RANDOM_SEED,
    )

    real_val, real_test = train_test_split(
        real_temp,
        test_size=0.50,
        stratify=real_temp["label"],
        random_state=RANDOM_SEED,
    )

    df.loc[real_train.index, "heldout_generator_split"] = "train"
    df.loc[real_val.index, "heldout_generator_split"] = "val"
    df.loc[real_test.index, "heldout_generator_split"] = "test"

    # AI split by generator
    ai_df = df[df["label"] == "ai_generated"].copy()

    generators = ai_df["generator_name"].dropna().unique().tolist()
    print("\nAvailable AI generators:")
    print(sorted(generators))

    if len(generators) < 2:
        raise ValueError(
            "Held-out generator split requires at least 2 distinct generator_name values "
            "in AI data."
        )

    generator_counts = ai_df["generator_name"].value_counts()

    if heldout_test_generator is None and heldout_val_generator is None:
        # Choose the two most frequent generators for stable val/test splits
        heldout_test_generator = generator_counts.index[0]
        heldout_val_generator = generator_counts.index[1]
    else:
        if heldout_test_generator is None or heldout_val_generator is None:
            raise ValueError("Please provide both heldout_test_generator and heldout_val_generator.")
        if heldout_test_generator == heldout_val_generator:
            raise ValueError("heldout_test_generator and heldout_val_generator must be different.")
        if heldout_test_generator not in generator_counts.index:
            raise ValueError(
                f"heldout_test_generator '{heldout_test_generator}' not found in generator_name values."
            )
        if heldout_val_generator not in generator_counts.index:
            raise ValueError(
                f"heldout_val_generator '{heldout_val_generator}' not found in generator_name values."
            )

    print(f"\nHeld-out test generator: {heldout_test_generator}")
    print(f"Held-out val generator: {heldout_val_generator}")

    ai_train = ai_df[
        ~ai_df["generator_name"].isin([heldout_test_generator, heldout_val_generator])
    ]
    ai_val = ai_df[ai_df["generator_name"] == heldout_val_generator]
    ai_test = ai_df[ai_df["generator_name"] == heldout_test_generator]

    df.loc[ai_train.index, "heldout_generator_split"] = "train"
    df.loc[ai_val.index, "heldout_generator_split"] = "val"
    df.loc[ai_test.index, "heldout_generator_split"] = "test"

    return df


def print_split_summary(df: pd.DataFrame):
    print("\n=== RANDOM SPLIT SUMMARY ===")
    print(pd.crosstab(df["random_split"], df["label"]))

    print("\n=== HELD-OUT GENERATOR SPLIT SUMMARY ===")
    print(pd.crosstab(df["heldout_generator_split"], df["label"]))

    print("\n=== HELD-OUT GENERATOR COUNTS (AI ONLY) ===")
    ai_df = df[df["label"] == "ai_generated"]
    print(pd.crosstab(ai_df["heldout_generator_split"], ai_df["generator_name"]))


def main():
    parser = argparse.ArgumentParser(description="Create dataset splits for RealVision.")
    parser.add_argument(
        "--heldout_test_generator",
        type=str,
        default=None,
        help="Generator name to hold out entirely for test (AI only).",
    )
    parser.add_argument(
        "--heldout_val_generator",
        type=str,
        default=None,
        help="Generator name to hold out entirely for val (AI only).",
    )
    args = parser.parse_args()

    df = pd.read_csv(INPUT_METADATA)

    df = create_random_split(df)
    df = create_heldout_generator_split(
        df,
        heldout_test_generator=args.heldout_test_generator,
        heldout_val_generator=args.heldout_val_generator,
    )

    df.to_csv(OUTPUT_METADATA, index=False)

    print(f"\nSaved metadata with splits to: {OUTPUT_METADATA}")
    print_split_summary(df)


if __name__ == "__main__":
    main()
