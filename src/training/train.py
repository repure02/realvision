from pathlib import Path
import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from src.training.dataset import create_dataloaders
from src.utils.config import get_training_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


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
def evaluate(model, loader, criterion, device, split_name="Val"):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    tp = 0
    fn = 0

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

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return epoch_loss, epoch_acc, recall_pos


def main():
    parser = argparse.ArgumentParser(description="Train a ResNet18 classifier.")
    parser.add_argument(
        "--split_column",
        type=str,
        default="heldout_generator_split",
        choices=["random_split", "heldout_generator_split"],
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    batch_size, num_workers, image_size, metadata_path = get_training_settings()

    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=metadata_path,
        split_column=args.split_column,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

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
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    num_epochs = args.epochs
    best_val_recall = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 3
    epochs_since_improve = 0

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

            if args.checkpoint_name:
                checkpoint_name = args.checkpoint_name
            else:
                checkpoint_name = f"resnet18_{args.split_column}_best.pt"
            checkpoint_path = CHECKPOINT_DIR / checkpoint_name
            torch.save(best_model_wts, checkpoint_path)
            print(f"Saved best checkpoint to: {checkpoint_path}")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping: no val recall improvement for {patience} epochs.")
                break

    model.load_state_dict(best_model_wts)

    test_loss, test_acc, test_recall = evaluate(
        model, test_loader, criterion, DEVICE, split_name="Test"
    )

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Best Val Recall: {best_val_recall:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Recall: {test_recall:.4f}")


if __name__ == "__main__":
    main()
