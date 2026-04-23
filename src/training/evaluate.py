from pathlib import Path
import argparse
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from torchvision import models
from tqdm import tqdm

from src.training.dataset import build_logo_splits, create_dataloaders_from_dfs
from src.utils.config import get_training_settings
from src.utils.logo_artifacts import logo_predictions_path, to_repo_relative
from src.utils.experiment_tracking import build_run_record, log_run, utc_now_iso

REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs_ai = []
    all_image_ids = []
    all_sources = []
    all_generators = []
    all_filepaths = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs_ai.extend(probs[:, 1].cpu().numpy().tolist())

        all_image_ids.extend(batch["image_id"])
        all_sources.extend(batch["source"])
        all_generators.extend(batch["generator_name"])
        all_filepaths.extend(batch["filepath"])

    results_df = pd.DataFrame(
        {
            "image_id": all_image_ids,
            "source": all_sources,
            "generator_name": all_generators,
            "filepath": all_filepaths,
            "true_label": all_labels,
            "pred_label": all_preds,
            "prob_ai": all_probs_ai,
        }
    )

    return results_df


def save_confusion_matrix(cm, output_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["real", "ai_generated"])
    plt.yticks([0, 1], ["real", "ai_generated"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_roc_curve(y_true, y_scores, output_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_pr_curve(y_true, y_scores, output_path, title):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_threshold_sweep(
    y_true,
    y_scores,
    thresholds,
    output_path,
):
    rows = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    sweep_df = pd.DataFrame(rows).sort_values("threshold")
    sweep_df.to_csv(output_path, index=False)
    return sweep_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ConvNeXt-Tiny LOGO checkpoint.")
    parser.add_argument(
        "--logo_test_generator",
        type=str,
        required=True,
        help="Generator held out as test for the LOGO checkpoint being evaluated.",
    )
    parser.add_argument(
        "--logo_val_generator",
        type=str,
        default=None,
        help="Optional validation generator override for reconstructing the LOGO split.",
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--threshold_sweep", action="store_true", help="Run threshold sweep on prob_ai.")
    parser.add_argument(
        "--target_recall",
        type=float,
        default=0.8,
        help="Target recall for selecting a threshold during sweep.",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=None,
        help="Override decision threshold for prob_ai (e.g., 0.10).",
    )
    args = parser.parse_args()
    started_at = utc_now_iso()

    print(f"Using device: {DEVICE}")

    batch_size, num_workers, image_size, metadata_path = get_training_settings()
    if metadata_path is None:
        raise FileNotFoundError("Training metadata path could not be resolved.")

    train_df, val_df, test_df, val_gen_used = build_logo_splits(
        metadata_path=metadata_path,
        heldout_test_generator=args.logo_test_generator,
        heldout_val_generator=args.logo_val_generator,
    )
    _, _, test_loader = create_dataloaders_from_dfs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
    else:
        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / f"convnext_tiny_logo_test_{args.logo_test_generator}_best.pt"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_tag = args.run_tag or f"logo_test_{args.logo_test_generator}"

    model = build_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    results_df = collect_predictions(model, test_loader, DEVICE)

    y_true = results_df["true_label"].values
    y_pred = results_df["pred_label"].values
    y_scores = results_df["prob_ai"].values

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== TEST METRICS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    if run_tag == f"logo_test_{args.logo_test_generator}":
        results_csv_path = logo_predictions_path(args.logo_test_generator)
    else:
        results_csv_path = PROJECT_ROOT / "reports" / f"{run_tag}_test_predictions.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved predictions to: {results_csv_path}")

    if args.decision_threshold is not None:
        thr = float(args.decision_threshold)
        y_pred_thr = (y_scores >= thr).astype(int)
        acc_thr = accuracy_score(y_true, y_pred_thr)
        precision_thr = precision_score(y_true, y_pred_thr, zero_division=0)
        recall_thr = recall_score(y_true, y_pred_thr, zero_division=0)
        f1_thr = f1_score(y_true, y_pred_thr, zero_division=0)
        cm_thr = confusion_matrix(y_true, y_pred_thr)

        print(f"\n=== THRESHOLD OVERRIDE (t={thr:.2f}) ===")
        print(f"Accuracy : {acc_thr:.4f}")
        print(f"Precision: {precision_thr:.4f}")
        print(f"Recall   : {recall_thr:.4f}")
        print(f"F1-score : {f1_thr:.4f}")
        print("\nConfusion Matrix:")
        print(cm_thr)

    save_confusion_matrix(
        cm,
        REPORTS_DIR / f"{run_tag}_confusion_matrix.png",
        title=f"{run_tag} Confusion Matrix",
    )
    save_roc_curve(
        y_true,
        y_scores,
        REPORTS_DIR / f"{run_tag}_roc_curve.png",
        title=f"{run_tag} ROC Curve",
    )
    save_pr_curve(
        y_true,
        y_scores,
        REPORTS_DIR / f"{run_tag}_pr_curve.png",
        title=f"{run_tag} Precision-Recall Curve",
    )

    if args.threshold_sweep:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
        sweep_path = PROJECT_ROOT / "reports" / f"{run_tag}_threshold_sweep.csv"
        sweep_df = run_threshold_sweep(y_true, y_scores, thresholds, sweep_path)
        target = args.target_recall
        eligible = sweep_df[sweep_df["recall"] >= target]
        if not eligible.empty:
            best_row = eligible.sort_values(["precision", "recall"], ascending=False).iloc[0]
            print("\n=== THRESHOLD SWEEP (TARGET RECALL) ===")
            print(
                f"Target recall: {target:.2f} | "
                f"Chosen threshold: {best_row['threshold']:.2f} | "
                f"Recall: {best_row['recall']:.4f} | "
                f"Precision: {best_row['precision']:.4f} | "
                f"Accuracy: {best_row['accuracy']:.4f} | "
                f"F1: {best_row['f1']:.4f}"
            )
        else:
            best_row = sweep_df.sort_values(["recall", "precision"], ascending=False).iloc[0]
            print("\n=== THRESHOLD SWEEP (TOP BY RECALL) ===")
            print(
                f"No threshold meets target recall {target:.2f}. "
                f"Best recall threshold: {best_row['threshold']:.2f} | "
                f"Recall: {best_row['recall']:.4f} | "
                f"Precision: {best_row['precision']:.4f} | "
                f"Accuracy: {best_row['accuracy']:.4f} | "
                f"F1: {best_row['f1']:.4f}"
            )

        chosen_path = PROJECT_ROOT / "reports" / f"{run_tag}_chosen_threshold.txt"
        with open(chosen_path, "w", encoding="utf-8") as f:
            f.write(
                f"threshold={best_row['threshold']:.2f}\n"
                f"recall={best_row['recall']:.4f}\n"
                f"precision={best_row['precision']:.4f}\n"
                f"accuracy={best_row['accuracy']:.4f}\n"
                f"f1={best_row['f1']:.4f}\n"
                f"target_recall={target:.2f}\n"
            )
        print(f"Saved threshold sweep to: {sweep_path}")
        print(f"Saved chosen threshold to: {chosen_path}")

    print(f"Saved figures to: {REPORTS_DIR}")

    metrics = {
        "test_acc": float(acc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_size": int(len(results_df)),
    }
    artifacts = {
        "predictions_csv": to_repo_relative(results_csv_path),
        "confusion_matrix_png": str((REPORTS_DIR / f"{run_tag}_confusion_matrix.png").relative_to(PROJECT_ROOT)),
        "roc_curve_png": str((REPORTS_DIR / f"{run_tag}_roc_curve.png").relative_to(PROJECT_ROOT)),
        "pr_curve_png": str((REPORTS_DIR / f"{run_tag}_pr_curve.png").relative_to(PROJECT_ROOT)),
    }
    if args.threshold_sweep:
        artifacts["threshold_sweep_csv"] = str(sweep_path.relative_to(PROJECT_ROOT))
        artifacts["chosen_threshold_txt"] = str(chosen_path.relative_to(PROJECT_ROOT))

    run_record = build_run_record(
        run_name=run_tag,
        task="evaluate",
        split_type="logo",
        args=args,
        metadata_path=metadata_path,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        artifacts=artifacts,
        extra={
            "model_name": "convnext_tiny",
            "device": str(DEVICE),
            "run_group": "logo_evaluation",
            "test_generator": args.logo_test_generator,
            "val_generator": val_gen_used,
        },
        started_at=started_at,
    )
    run_path = log_run(run_record)
    print(f"Logged run metadata to: {run_path}")


if __name__ == "__main__":
    main()
