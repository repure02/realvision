from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms

from src.utils.config import get_training_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FINAL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "convnext_tiny_final_inference_best.pt"
FINAL_THRESHOLD_PATH = REPORTS_DIR / "final_inference_chosen_threshold.txt"


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


def get_eval_transform(image_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_threshold_from_file(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip().splitlines()
        for line in content:
            if line.startswith("threshold="):
                return float(line.split("=", 1)[1])
    except Exception:
        return None
    return None


def list_available_logo_generators() -> list[str]:
    summary_path = REPORTS_DIR / "logo_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if "test_generator" in df.columns:
            values = sorted(df["test_generator"].dropna().astype(str).unique().tolist())
            if values:
                return values

    prefix = "convnext_tiny_logo_test_"
    suffix = "_best.pt"
    generators = []
    for path in CHECKPOINTS_DIR.glob("convnext_tiny_logo_test_*_best.pt"):
        name = path.name
        generators.append(name[len(prefix):-len(suffix)])
    return sorted(set(generators))


def resolve_model_selection(
    model_mode: str,
    checkpoint_path_arg: str | None,
    logo_test_generator: str | None,
) -> tuple[Path, str, float | None]:
    if checkpoint_path_arg:
        checkpoint_path = Path(checkpoint_path_arg)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        return checkpoint_path, "custom", None

    use_final = model_mode in {"auto", "final"} and FINAL_CHECKPOINT_PATH.exists()
    if model_mode == "final" and not FINAL_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Final inference checkpoint not found: {FINAL_CHECKPOINT_PATH}")
    if use_final:
        return FINAL_CHECKPOINT_PATH, "final_inference", load_threshold_from_file(FINAL_THRESHOLD_PATH)

    available_generators = list_available_logo_generators()
    if not available_generators:
        raise FileNotFoundError(
            "No LOGO checkpoints found. Generate LOGO checkpoints or the final inference checkpoint first."
        )
    selected_generator = logo_test_generator or available_generators[0]
    checkpoint_path = CHECKPOINTS_DIR / f"convnext_tiny_logo_test_{selected_generator}_best.pt"
    return checkpoint_path, f"logo:{selected_generator}", None


@torch.no_grad()
def predict_image(
    image_path: Path,
    checkpoint_path: Path,
    image_size: int,
    decision_threshold: float | None,
) -> dict:
    model = build_model(num_classes=2).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    transform = get_eval_transform(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    prob_real = float(probs[0])
    prob_ai = float(probs[1])

    if decision_threshold is None:
        pred_label = "ai_generated" if prob_ai >= 0.5 else "real"
        threshold_used = 0.5
    else:
        pred_label = "ai_generated" if prob_ai >= decision_threshold else "real"
        threshold_used = decision_threshold

    return {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "prob_ai": prob_ai,
        "prob_real": prob_real,
        "pred_label": pred_label,
        "decision_threshold": threshold_used,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the final model or a selected LOGO checkpoint.")
    parser.add_argument("image_path", type=str, help="Path to input image.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a specific checkpoint. Overrides model-mode selection.",
    )
    parser.add_argument(
        "--model_mode",
        type=str,
        default="auto",
        choices=["auto", "final", "logo"],
        help="Prefer the final inference checkpoint when available, or force final/logo behavior.",
    )
    parser.add_argument(
        "--logo_test_generator",
        type=str,
        default=None,
        help="Held-out generator name used to select checkpoints/convnext_tiny_logo_test_<generator>_best.pt in logo mode.",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=None,
        help="Override decision threshold for prob_ai. In final mode, the saved calibrated threshold is used by default when present.",
    )
    args = parser.parse_args()

    _, _, image_size, _ = get_training_settings()

    image_path = Path(args.image_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint_path, model_label, saved_threshold = resolve_model_selection(
        model_mode=args.model_mode,
        checkpoint_path_arg=args.checkpoint_path,
        logo_test_generator=args.logo_test_generator,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    decision_threshold = args.decision_threshold if args.decision_threshold is not None else saved_threshold

    print(f"Using device: {DEVICE}")
    print(f"Image size: {image_size}")
    print(f"Model selection: {model_label}")

    result = predict_image(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        decision_threshold=decision_threshold,
    )

    print("\n=== PREDICTION ===")
    print(f"Image: {result['image_path']}")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Prob AI: {result['prob_ai']:.4f}")
    print(f"Prob Real: {result['prob_real']:.4f}")
    print(f"Decision Threshold: {result['decision_threshold']:.2f}")
    print(f"Predicted: {result['pred_label']}")
    print("Decision rule: predict AI if prob_ai >= threshold, else real.")


if __name__ == "__main__":
    main()
