from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from src.training.dataset import build_logo_splits, create_dataloaders_from_dfs
from src.training.train import _collect_predictions
from src.utils.config import get_training_settings
from src.utils.logo_artifacts import (
    PROJECT_ROOT,
    logo_generator_metrics_path,
    logo_predictions_path,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def build_model(num_classes: int = 2):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def save_generator_metrics(preds_df: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for gen, group in preds_df.groupby("generator_name", dropna=False):
        tp = int(((group["true_label"] == 1) & (group["pred_label"] == 1)).sum())
        fp = int(((group["true_label"] == 0) & (group["pred_label"] == 1)).sum())
        fn = int(((group["true_label"] == 1) & (group["pred_label"] == 0)).sum())
        tn = int(((group["true_label"] == 0) & (group["pred_label"] == 0)).sum())
        count = int(len(group))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        acc = (tp + tn) / count if count > 0 else 0.0
        rows.append(
            {
                "generator_name": str(gen),
                "count": count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "accuracy": round(acc, 6),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("generator_name").to_csv(output_path, index=False)


def run_backfill(summary_path: Path, missing_only: bool) -> None:
    batch_size, num_workers, image_size, metadata_path = get_training_settings()
    if metadata_path is None:
        raise FileNotFoundError("Training metadata path could not be resolved.")

    summary_df = pd.read_csv(summary_path)
    required = {"test_generator", "val_generator"}
    if not required.issubset(summary_df.columns):
        raise ValueError(f"{summary_path} must include columns: {sorted(required)}")

    for _, row in summary_df.iterrows():
        test_generator = str(row["test_generator"])
        val_generator = str(row["val_generator"])

        pred_path = logo_predictions_path(test_generator)
        metrics_path = logo_generator_metrics_path(test_generator)
        if missing_only and pred_path.exists() and metrics_path.exists():
            print(f"Skipping {test_generator}: detail files already exist.")
            continue

        checkpoint_path = (
            PROJECT_ROOT / "checkpoints" / f"convnext_tiny_logo_test_{test_generator}_best.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {test_generator}: {checkpoint_path}")

        print(f"Backfilling LOGO details for {test_generator} using val generator {val_generator}.")

        train_df, val_df, test_df, _ = build_logo_splits(
            metadata_path=metadata_path,
            heldout_test_generator=test_generator,
            heldout_val_generator=val_generator,
        )
        _, _, test_loader = create_dataloaders_from_dfs(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        model = build_model(num_classes=2).to(DEVICE)
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        preds_df = _collect_predictions(model, test_loader, DEVICE, split_name=test_generator)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(pred_path, index=False)
        save_generator_metrics(preds_df, metrics_path)

        print(f"Saved: {pred_path}")
        print(f"Saved: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing LOGO detail artifacts from existing checkpoints."
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="reports/logo_summary.csv",
        help="Path to the canonical LOGO summary CSV.",
    )
    parser.add_argument(
        "--missing_only",
        action="store_true",
        help="Only generate files that are currently missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    run_backfill(summary_path=summary_path, missing_only=args.missing_only)


if __name__ == "__main__":
    main()
