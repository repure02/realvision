from pathlib import Path
import argparse
import copy
import warnings

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm

import pandas as pd

from src.training.dataset import (
    build_final_inference_splits,
    build_logo_splits,
    create_dataloaders_from_dfs,
    print_split_summary,
)
from src.utils.config import get_training_settings
from src.utils.logo_artifacts import (
    REPORTS_DIR,
    logo_generator_metrics_path,
    logo_predictions_path,
    to_repo_relative,
)
from src.utils.experiment_tracking import build_run_record, log_run, utc_now_iso


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_INFERENCE_CHECKPOINT_NAME = "convnext_tiny_final_inference_best.pt"
FINAL_INFERENCE_PREDICTIONS_PATH = REPORTS_DIR / "final_inference_calibration_predictions.csv"
FINAL_INFERENCE_GENERATOR_METRICS_PATH = REPORTS_DIR / "final_inference_generator_metrics.csv"
FINAL_INFERENCE_SWEEP_PATH = REPORTS_DIR / "final_inference_threshold_sweep.csv"
FINAL_INFERENCE_THRESHOLD_PATH = REPORTS_DIR / "final_inference_chosen_threshold.txt"

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
    try:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    except Exception as exc:
        warnings.warn(
            "Could not load pretrained ConvNeXt-Tiny weights. "
            "Falling back to random initialization, which is safer for offline Colab runs. "
            f"Original error: {exc}",
            RuntimeWarning,
        )
        model = models.convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss
        return loss.mean()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="Val", return_generator_stats=False):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    tp = 0
    fn = 0
    generator_stats = {} if return_generator_stats else None

    for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        if return_generator_stats:
            preds_cpu = preds.detach().cpu().tolist()
            labels_cpu = labels.detach().cpu().tolist()
            gen_cpu = [str(g) for g in batch["generator_name"]]
            for pred, label, gen in zip(preds_cpu, labels_cpu, gen_cpu):
                stats = generator_stats.setdefault(
                    gen, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "count": 0}
                )
                stats["count"] += 1
                if label == 1 and pred == 1:
                    stats["tp"] += 1
                elif label == 0 and pred == 1:
                    stats["fp"] += 1
                elif label == 1 and pred == 0:
                    stats["fn"] += 1
                else:
                    stats["tn"] += 1

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if return_generator_stats:
        return epoch_loss, epoch_acc, recall_pos, generator_stats
    return epoch_loss, epoch_acc, recall_pos


def _save_generator_metrics(generator_stats: dict, output_path: Path) -> None:
    rows = []
    for gen, stats in sorted(generator_stats.items()):
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        tn = stats["tn"]
        count = stats["count"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        acc = (tp + tn) / count if count > 0 else 0.0
        rows.append(
            {
                "generator_name": gen,
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
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _generator_metrics_from_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gen, group in predictions_df.groupby("generator_name", sort=True):
        y_true = group["true_label"].astype(int)
        y_pred = group["pred_label"].astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        count = int(len(group))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        acc = (tp + tn) / count if count > 0 else 0.0
        rows.append(
            {
                "generator_name": gen,
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
    return pd.DataFrame(rows)


def _write_thresholded_prediction_artifacts(
    predictions_df: pd.DataFrame,
    predictions_path: Path,
    generator_metrics_path: Path,
    threshold: float,
) -> pd.DataFrame:
    output_df = predictions_df.copy()
    if "pred_label_argmax" not in output_df.columns:
        output_df["pred_label_argmax"] = output_df["pred_label"].astype(int)
    output_df["decision_threshold"] = float(threshold)
    output_df["pred_label"] = (output_df["prob_ai"].astype(float) >= threshold).astype(int)

    preferred_columns = [
        "image_id",
        "source",
        "generator_name",
        "filepath",
        "true_label",
        "pred_label",
        "pred_label_argmax",
        "decision_threshold",
        "prob_ai",
    ]
    ordered_columns = [col for col in preferred_columns if col in output_df.columns]
    ordered_columns.extend(col for col in output_df.columns if col not in ordered_columns)
    output_df = output_df[ordered_columns]

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(predictions_path, index=False)

    generator_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _generator_metrics_from_predictions(output_df).to_csv(generator_metrics_path, index=False)
    return output_df


@torch.no_grad()
def _collect_predictions(model, loader, device, split_name: str) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in tqdm(loader, desc=f"Predicting {split_name}", leave=False):
        images = batch["image"].to(device)
        outputs = model(images)
        probs_ai = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().tolist()
        preds = outputs.argmax(dim=1).detach().cpu().tolist()
        labels = batch["label"].detach().cpu().tolist()

        image_ids = [str(x) for x in batch["image_id"]]
        sources = [str(x) for x in batch["source"]]
        generators = [str(x) for x in batch["generator_name"]]
        filepaths = [str(x) for x in batch["filepath"]]

        for image_id, source, gen, path, label, pred, prob in zip(
            image_ids, sources, generators, filepaths, labels, preds, probs_ai
        ):
            rows.append(
                {
                    "image_id": image_id,
                    "source": source,
                    "generator_name": gen,
                    "filepath": path,
                    "true_label": int(label),
                    "pred_label": int(pred),
                    "prob_ai": float(prob),
                }
            )
    return pd.DataFrame(rows)


def _write_threshold_artifacts(
    predictions_df: pd.DataFrame,
    sweep_path: Path,
    chosen_path: Path,
    target_recall: float,
) -> dict:
    y_true = predictions_df["true_label"].astype(int).values
    y_scores = predictions_df["prob_ai"].astype(float).values

    rows = []
    thresholds = [round(x, 2) for x in [i / 100 for i in range(5, 96, 5)]]
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )

    sweep_df = pd.DataFrame(rows).sort_values("threshold")
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(sweep_path, index=False)

    eligible = sweep_df[sweep_df["recall"] >= target_recall]
    if not eligible.empty:
        chosen = eligible.sort_values(["precision", "recall"], ascending=False).iloc[0]
    else:
        chosen = sweep_df.sort_values(["recall", "precision"], ascending=False).iloc[0]

    chosen_metrics = {
        "threshold": float(chosen["threshold"]),
        "accuracy": float(chosen["accuracy"]),
        "precision": float(chosen["precision"]),
        "recall": float(chosen["recall"]),
        "f1": float(chosen["f1"]),
        "target_recall": float(target_recall),
    }
    chosen_path.write_text(
        (
            f"threshold={chosen_metrics['threshold']:.2f}\n"
            f"recall={chosen_metrics['recall']:.4f}\n"
            f"precision={chosen_metrics['precision']:.4f}\n"
            f"accuracy={chosen_metrics['accuracy']:.4f}\n"
            f"f1={chosen_metrics['f1']:.4f}\n"
            f"target_recall={chosen_metrics['target_recall']:.2f}\n"
        ),
        encoding="utf-8",
    )
    return chosen_metrics


def _run_training_once(
    train_loader,
    val_loader,
    test_loader,
    args,
    split_label: str,
    metadata_path: Path | None,
    checkpoint_name_override: str | None = None,
    generator_metrics_path: Path | None = None,
    test_predictions_path: Path | None = None,
    extra_tracking: dict | None = None,
):
    started_at = utc_now_iso()
    model = build_model(num_classes=2).to(DEVICE)

    lr = 3e-5
    weight_decay = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Rebuild criterion with class weights to favor recall on AI class (1).
    train_labels = train_loader.dataset.df["label"].map(train_loader.dataset.label_to_index).values
    total = len(train_labels)
    count_0 = int((train_labels == 0).sum())
    count_1 = int((train_labels == 1).sum())
    w0 = total / (2 * max(count_0, 1))
    w1 = total / (2 * max(count_1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float, device=DEVICE)
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    num_epochs = args.epochs
    best_val_recall = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 3
    epochs_since_improve = 0
    metric_key = "best_val_recall"

    if checkpoint_name_override:
        checkpoint_name = checkpoint_name_override
    elif args.checkpoint_name:
        checkpoint_name = args.checkpoint_name
    else:
        checkpoint_name = f"convnext_tiny_{split_label}_best.pt"
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    global_best = None

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc, val_recall = evaluate(
            model, val_loader, criterion, DEVICE, split_name="Val"
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val Recall: {val_recall:.4f}")

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0

            should_save_global = global_best is None or val_recall > global_best
            if should_save_global:
                torch.save(best_model_wts, checkpoint_path)
                global_best = val_recall
                print(f"Saved best checkpoint to: {checkpoint_path}")
            else:
                print(
                    f"Best in this run improved to {val_recall:.4f}, "
                    f"but did not beat previous best {global_best:.4f}. "
                    "Checkpoint not overwritten."
                )
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping: no val recall improvement for {patience} epochs.")
                break

    model.load_state_dict(best_model_wts)

    if generator_metrics_path:
        test_loss, test_acc, test_recall, generator_stats = evaluate(
            model, test_loader, criterion, DEVICE, split_name="Test", return_generator_stats=True
        )
        _save_generator_metrics(generator_stats, generator_metrics_path)
    else:
        test_loss, test_acc, test_recall = evaluate(
            model, test_loader, criterion, DEVICE, split_name="Test"
        )

    if test_predictions_path:
        preds_df = _collect_predictions(model, test_loader, DEVICE, split_name="Test")
        test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(test_predictions_path, index=False)

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Best Val Recall: {best_val_recall:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    metrics = {
        "best_val_recall": best_val_recall,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_recall": test_recall,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "epochs_requested": args.epochs,
        "epochs_completed": epoch + 1,
    }
    artifacts = {}
    if generator_metrics_path is not None:
        artifacts["generator_metrics_csv"] = to_repo_relative(generator_metrics_path)
    if test_predictions_path is not None:
        artifacts["test_predictions_csv"] = to_repo_relative(test_predictions_path)

    run_record = build_run_record(
        run_name=split_label,
        task="train",
        split_type="final_inference" if getattr(args, "final_inference", False) else "logo",
        args=args,
        metadata_path=metadata_path,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        artifacts=artifacts,
        extra={
            "model_name": "convnext_tiny",
            "device": str(DEVICE),
            "run_group": "final_inference" if getattr(args, "final_inference", False) else "logo",
            **(extra_tracking or {}),
        },
        started_at=started_at,
    )
    run_path = log_run(run_record)
    print(f"Logged run metadata to: {run_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-Tiny checkpoints for LOGO benchmarking or the final inference model."
    )
    parser.add_argument(
        "--final_inference",
        action="store_true",
        help="Train the single deployment checkpoint on all known generators with a held-out calibration split.",
    )
    parser.add_argument(
        "--logo_all",
        action="store_true",
        help="Run LOGO over all generators and report aggregate metrics.",
    )
    parser.add_argument(
        "--logo_test_generator",
        type=str,
        default=None,
        help="Generator name to hold out as test in LOGO mode.",
    )
    parser.add_argument(
        "--logo_val_generator",
        type=str,
        default=None,
        help="Generator name to hold out as val in LOGO mode (optional).",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Loss function: standard cross-entropy (ce) or focal loss (focal).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (only used when --loss=focal).",
    )
    parser.add_argument(
        "--final_val_fraction",
        type=float,
        default=0.15,
        help="Validation fraction carved from the random train/val pool for the final inference model.",
    )
    parser.add_argument(
        "--target_recall",
        type=float,
        default=0.8,
        help="Target recall used when choosing the saved decision threshold for the final inference model.",
    )
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    batch_size, num_workers, image_size, metadata_path = get_training_settings()
    if metadata_path is None:
        raise ValueError("LOGO training requires a metadata path.")

    df = pd.read_csv(metadata_path)
    if "generator_name" not in df.columns or "label" not in df.columns:
        raise ValueError("Metadata must include 'generator_name' and 'label' for LOGO training.")
    generators = sorted(df[df["label"] == "ai_generated"]["generator_name"].dropna().unique())
    if not generators:
        raise ValueError("No AI generators found in metadata for LOGO training.")

    if args.final_inference:
        train_df, val_df, calibration_df = build_final_inference_splits(
            metadata_path=metadata_path,
            val_fraction=args.final_val_fraction,
        )
        print_split_summary(train_df, "Train")
        print_split_summary(val_df, "Val")
        print_split_summary(calibration_df, "Calibration")

        train_loader, val_loader, calibration_loader = create_dataloaders_from_dfs(
            train_df=train_df,
            val_df=val_df,
            test_df=calibration_df,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        _run_training_once(
            train_loader,
            val_loader,
            calibration_loader,
            args,
            split_label="final_inference",
            metadata_path=metadata_path,
            checkpoint_name_override=FINAL_INFERENCE_CHECKPOINT_NAME,
            generator_metrics_path=FINAL_INFERENCE_GENERATOR_METRICS_PATH,
            test_predictions_path=FINAL_INFERENCE_PREDICTIONS_PATH,
            extra_tracking={
                "calibration_split": "random_split:test",
                "final_val_fraction": args.final_val_fraction,
            },
        )

        predictions_df = pd.read_csv(FINAL_INFERENCE_PREDICTIONS_PATH)
        chosen_metrics = _write_threshold_artifacts(
            predictions_df=predictions_df,
            sweep_path=FINAL_INFERENCE_SWEEP_PATH,
            chosen_path=FINAL_INFERENCE_THRESHOLD_PATH,
            target_recall=args.target_recall,
        )
        predictions_df = _write_thresholded_prediction_artifacts(
            predictions_df=predictions_df,
            predictions_path=FINAL_INFERENCE_PREDICTIONS_PATH,
            generator_metrics_path=FINAL_INFERENCE_GENERATOR_METRICS_PATH,
            threshold=chosen_metrics["threshold"],
        )
        print("\n=== FINAL INFERENCE THRESHOLD ===")
        print(
            f"Chosen threshold: {chosen_metrics['threshold']:.2f} | "
            f"Recall: {chosen_metrics['recall']:.4f} | "
            f"Precision: {chosen_metrics['precision']:.4f} | "
            f"Accuracy: {chosen_metrics['accuracy']:.4f} | "
            f"F1: {chosen_metrics['f1']:.4f}"
        )
        print(f"Updated calibration predictions with thresholded labels: {FINAL_INFERENCE_PREDICTIONS_PATH}")
        print(f"Updated thresholded generator metrics: {FINAL_INFERENCE_GENERATOR_METRICS_PATH}")
        print(f"Saved threshold sweep to: {FINAL_INFERENCE_SWEEP_PATH}")
        print(f"Saved chosen threshold to: {FINAL_INFERENCE_THRESHOLD_PATH}")

        calibration_record = build_run_record(
            run_name="final_inference_calibration",
            task="calibrate",
            split_type="final_inference",
            args=args,
            metadata_path=metadata_path,
            checkpoint_path=CHECKPOINT_DIR / FINAL_INFERENCE_CHECKPOINT_NAME,
            metrics={
                "chosen_threshold": chosen_metrics["threshold"],
                "calibration_acc": chosen_metrics["accuracy"],
                "calibration_precision": chosen_metrics["precision"],
                "calibration_recall": chosen_metrics["recall"],
                "calibration_f1": chosen_metrics["f1"],
            },
            artifacts={
                "predictions_csv": to_repo_relative(FINAL_INFERENCE_PREDICTIONS_PATH),
                "generator_metrics_csv": to_repo_relative(FINAL_INFERENCE_GENERATOR_METRICS_PATH),
                "threshold_sweep_csv": to_repo_relative(FINAL_INFERENCE_SWEEP_PATH),
                "chosen_threshold_txt": to_repo_relative(FINAL_INFERENCE_THRESHOLD_PATH),
            },
            extra={
                "model_name": "convnext_tiny",
                "device": str(DEVICE),
                "run_group": "final_inference",
                "calibration_split": "random_split:test",
            },
        )
        run_path = log_run(calibration_record)
        print(f"Logged calibration metadata to: {run_path}")
        return

    if args.logo_all:
        results = []
        for gen in generators:
            val_gen = args.logo_val_generator
            if val_gen == gen:
                val_gen = None
            print(f"\n=== LOGO: test generator = {gen} ===")
            train_df, val_df, test_df, val_gen_used = build_logo_splits(
                metadata_path=metadata_path,
                heldout_test_generator=gen,
                heldout_val_generator=val_gen,
            )
            print_split_summary(train_df, "Train")
            print_split_summary(val_df, "Val")
            print_split_summary(test_df, "Test")
            train_loader, val_loader, test_loader = create_dataloaders_from_dfs(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            run_id = f"logo_test_{gen}"
            generator_metrics_path = logo_generator_metrics_path(gen)
            test_predictions_path = logo_predictions_path(gen)
            metrics = _run_training_once(
                train_loader,
                val_loader,
                test_loader,
                args,
                split_label=run_id,
                metadata_path=metadata_path,
                checkpoint_name_override=f"convnext_tiny_{run_id}_best.pt",
                generator_metrics_path=generator_metrics_path,
                test_predictions_path=test_predictions_path,
                extra_tracking={
                    "test_generator": gen,
                    "val_generator": val_gen_used,
                },
            )
            metrics["test_generator"] = gen
            metrics["val_generator"] = val_gen_used
            metrics["generator_metrics_csv"] = to_repo_relative(generator_metrics_path)
            metrics["test_predictions_csv"] = to_repo_relative(test_predictions_path)
            results.append(metrics)

        if results:
            summary_path = REPORTS_DIR / "logo_summary.csv"
            pd.DataFrame(results).to_csv(summary_path, index=False)
            avg_test_recall = sum(r["test_recall"] for r in results) / len(results)
            avg_test_acc = sum(r["test_acc"] for r in results) / len(results)
            print("\n=== LOGO AGGREGATE RESULTS ===")
            print(f"Average Test Acc: {avg_test_acc:.4f}")
            print(f"Average Test Recall: {avg_test_recall:.4f}")
            print(f"Saved LOGO summary to: {summary_path}")
        return

    if not args.logo_test_generator:
        raise ValueError("Provide --logo_test_generator for a single LOGO run or use --logo_all.")

    train_df, val_df, test_df, val_gen_used = build_logo_splits(
        metadata_path=metadata_path,
        heldout_test_generator=args.logo_test_generator,
        heldout_val_generator=args.logo_val_generator,
    )
    print_split_summary(train_df, "Train")
    print_split_summary(val_df, "Val")
    print_split_summary(test_df, "Test")
    train_loader, val_loader, test_loader = create_dataloaders_from_dfs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    run_id = f"logo_test_{args.logo_test_generator}"
    generator_metrics_path = logo_generator_metrics_path(args.logo_test_generator)
    test_predictions_path = logo_predictions_path(args.logo_test_generator)
    _run_training_once(
        train_loader,
        val_loader,
        test_loader,
        args,
        split_label=run_id,
        metadata_path=metadata_path,
        checkpoint_name_override=f"convnext_tiny_{run_id}_best.pt",
        generator_metrics_path=generator_metrics_path,
        test_predictions_path=test_predictions_path,
        extra_tracking={
            "test_generator": args.logo_test_generator,
            "val_generator": val_gen_used,
        },
    )


if __name__ == "__main__":
    main()
