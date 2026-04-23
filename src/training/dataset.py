from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils.config import get_env_or_config, load_dataset_config, get_paths_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "processed_metadata.csv"


def resolve_metadata_path() -> Path:
    config = load_dataset_config()
    env_path = get_env_or_config("REALVISION_METADATA_PATH", config)
    if env_path:
        path = Path(env_path)
        return path if path.is_absolute() else PROJECT_ROOT / path
    paths_config = get_paths_config(config)
    metadata_csv = paths_config.get("processed_metadata_csv")
    if metadata_csv and metadata_csv.exists():
        return metadata_csv
    metadata_csv = paths_config.get("master_metadata_csv")
    if metadata_csv and metadata_csv.exists():
        return metadata_csv
    if DEFAULT_METADATA_PATH.exists():
        return DEFAULT_METADATA_PATH
    return DEFAULT_METADATA_PATH


class RealVisionDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = metadata_df.reset_index(drop=True).copy()
        self.df["image_id"] = self.df["image_id"].astype(str)
        self.df["source"] = self.df["source"].astype(str)

        if "processed_filepath" in self.df.columns:
            self.filepath_column = "processed_filepath"
        elif "filepath" in self.df.columns:
            self.filepath_column = "filepath"
        else:
            raise ValueError("Metadata must include either 'processed_filepath' or 'filepath'.")

        self.df[self.filepath_column] = self.df[self.filepath_column].astype(str)

        if "generator_name" in self.df.columns:
            self.df["generator_name"] = self.df["generator_name"].fillna("none").astype(str)
        else:
            self.df["generator_name"] = "none"

        self.transform = transform

        self.label_to_index = {
            "real": 0,
            "ai_generated": 1,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = PROJECT_ROOT / row[self.filepath_column]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label_to_index[row["label"]]
        label = torch.tensor(label, dtype=torch.long)

        generator_name = row["generator_name"] if pd.notna(row["generator_name"]) else "none"

        sample = {
            "image": image,
            "label": label,
            "image_id": str(row["image_id"]),
            "source": str(row["source"]),
            "generator_name": str(generator_name),
            "filepath": str(row[self.filepath_column]),
        }

        return sample


def get_default_transforms(image_size: int = 224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
            p=0.5,
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def _load_metadata_df(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)

    columns_to_drop = {
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
    }

    drop_list = [col for col in df.columns if col.lower() in columns_to_drop]
    if drop_list:
        df = df.drop(columns=drop_list)

    return df


def load_split_dataframe(
    metadata_path: Path,
    split_column: str,
    split_value: str,
) -> pd.DataFrame:
    df = _load_metadata_df(metadata_path)

    if split_column not in df.columns:
        raise ValueError(f"Split column '{split_column}' not found in metadata.")

    split_df = df[df[split_column] == split_value].copy()

    if len(split_df) == 0:
        raise ValueError(
            f"No rows found for split_column='{split_column}', split_value='{split_value}'."
        )

    return split_df


def build_logo_splits(
    metadata_path: Path,
    heldout_test_generator: str,
    heldout_val_generator: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    df = _load_metadata_df(metadata_path)

    if "generator_name" not in df.columns:
        raise ValueError("Metadata must include 'generator_name' for LOGO splits.")
    if "label" not in df.columns:
        raise ValueError("Metadata must include 'label' for LOGO splits.")
    if "random_split" not in df.columns:
        raise ValueError("Metadata must include 'random_split' for LOGO splits.")

    ai_df = df[df["label"] == "ai_generated"].copy()
    real_df = df[df["label"] == "real"].copy()

    available_generators = sorted(set(ai_df["generator_name"].dropna().astype(str)))
    if heldout_test_generator not in available_generators:
        raise ValueError(
            f"heldout_test_generator='{heldout_test_generator}' not found in metadata."
        )

    remaining = [g for g in available_generators if g != heldout_test_generator]
    if heldout_val_generator is None:
        if not remaining:
            raise ValueError("Not enough generators to create a validation split.")
        # Pick the most frequent remaining generator for a stable val set.
        counts = ai_df[ai_df["generator_name"].isin(remaining)]["generator_name"].value_counts()
        heldout_val_generator = counts.index[0]
    elif heldout_val_generator not in remaining:
        raise ValueError(
            "heldout_val_generator must be different from heldout_test_generator and exist in metadata."
        )

    ai_test = ai_df[ai_df["generator_name"] == heldout_test_generator].copy()
    ai_val = ai_df[ai_df["generator_name"] == heldout_val_generator].copy()
    ai_train = ai_df[
        ~ai_df["generator_name"].isin([heldout_test_generator, heldout_val_generator])
    ].copy()

    real_train = real_df[real_df["random_split"] == "train"].copy()
    real_val = real_df[real_df["random_split"] == "val"].copy()
    real_test = real_df[real_df["random_split"] == "test"].copy()

    train_df = pd.concat([ai_train, real_train], ignore_index=True)
    val_df = pd.concat([ai_val, real_val], ignore_index=True)
    test_df = pd.concat([ai_test, real_test], ignore_index=True)

    return train_df, val_df, test_df, heldout_val_generator


def _build_stratify_labels(df: pd.DataFrame) -> pd.Series:
    generator_series = df.get("generator_name", pd.Series(index=df.index, dtype="object"))
    generator_series = generator_series.fillna("none").astype(str)
    labels = df["label"].astype(str)
    stratify = labels.where(labels != "ai_generated", labels + "::" + generator_series)
    value_counts = stratify.value_counts()
    if (value_counts < 2).any():
        return labels
    return stratify


def build_final_inference_splits(
    metadata_path: Path,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.05 <= val_fraction <= 0.40:
        raise ValueError("val_fraction must be between 0.05 and 0.40 for final inference splits.")

    df = _load_metadata_df(metadata_path)
    if "random_split" not in df.columns:
        raise ValueError("Metadata must include 'random_split' for final inference splits.")
    if "label" not in df.columns:
        raise ValueError("Metadata must include 'label' for final inference splits.")

    development_df = df[df["random_split"].isin(["train", "val"])].copy()
    calibration_df = df[df["random_split"] == "test"].copy()

    if development_df.empty or calibration_df.empty:
        raise ValueError(
            "Final inference splits require non-empty development rows (train/val) and "
            "calibration rows (test) in random_split."
        )

    stratify_labels = _build_stratify_labels(development_df)
    train_df, val_df = train_test_split(
        development_df,
        test_size=val_fraction,
        stratify=stratify_labels,
        random_state=42,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        calibration_df.reset_index(drop=True),
    )


def print_split_summary(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        print(f"{name} split is empty.")
        return
    label_counts = df["label"].value_counts().to_dict()
    print(f"{name} split size: {len(df)} | labels: {label_counts}")
    if "generator_name" in df.columns:
        ai_only = df[df["label"] == "ai_generated"]
        if not ai_only.empty:
            gen_counts = ai_only["generator_name"].value_counts().head(10).to_dict()
            print(f"{name} top AI generators: {gen_counts}")


def create_dataloaders(
    metadata_path: Path | None = None,
    split_column: str = "random_split",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
):
    if metadata_path is None:
        metadata_path = resolve_metadata_path()
    train_df = load_split_dataframe(metadata_path, split_column, "train")
    val_df = load_split_dataframe(metadata_path, split_column, "val")
    test_df = load_split_dataframe(metadata_path, split_column, "test")

    train_transform, eval_transform = get_default_transforms(image_size=image_size)

    train_dataset = RealVisionDataset(train_df, transform=train_transform)
    val_dataset = RealVisionDataset(val_df, transform=eval_transform)
    test_dataset = RealVisionDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader


def create_dataloaders_from_dfs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_transform, eval_transform = get_default_transforms(image_size=image_size)

    train_dataset = RealVisionDataset(train_df, transform=train_transform)
    val_dataset = RealVisionDataset(val_df, transform=eval_transform)
    test_dataset = RealVisionDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    batch = next(iter(train_loader))

    print("\nBatch keys:", batch.keys())
    print("Image batch shape:", batch["image"].shape)
    print("Label batch shape:", batch["label"].shape)
    print("First 5 labels:", batch["label"][:5])
    print("First 5 image_ids:", batch["image_id"][:5])
