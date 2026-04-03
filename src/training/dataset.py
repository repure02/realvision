from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils.config import get_env_or_config, load_dataset_config, get_paths_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "processed_metadata_with_splits.csv"
ALT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "processed_metadata_with_splits_224.csv"


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
    if ALT_METADATA_PATH.exists():
        return ALT_METADATA_PATH
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
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
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


def load_split_dataframe(
    metadata_path: Path,
    split_column: str,
    split_value: str,
) -> pd.DataFrame:
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
        "processed_filepath",
    }

    drop_list = [col for col in df.columns if col.lower() in columns_to_drop]
    if drop_list:
        df = df.drop(columns=drop_list)

    if split_column not in df.columns:
        raise ValueError(f"Split column '{split_column}' not found in metadata.")

    split_df = df[df[split_column] == split_value].copy()

    if len(split_df) == 0:
        raise ValueError(
            f"No rows found for split_column='{split_column}', split_value='{split_value}'."
        )

    return split_df


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
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
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
