from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

try:
    from src.utils.config import get_training_settings
except ModuleNotFoundError:
    alt_root = Path(__file__).resolve().parents[1]
    if str(alt_root) not in sys.path:
        sys.path.insert(0, str(alt_root))
    try:
        from src.utils.config import get_training_settings
    except ModuleNotFoundError:
        get_training_settings = None

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


def get_model_options() -> tuple[dict[str, tuple[Path, float | None]], str]:
    options: dict[str, tuple[Path, float | None]] = {}
    default_label = ""
    if FINAL_CHECKPOINT_PATH.exists():
        options["Final Inference Model"] = (
            FINAL_CHECKPOINT_PATH,
            load_threshold_from_file(FINAL_THRESHOLD_PATH),
        )
        default_label = "Final Inference Model"

    for generator in list_available_logo_generators():
        options[f"LOGO Benchmark: {generator}"] = (
            CHECKPOINTS_DIR / f"convnext_tiny_logo_test_{generator}_best.pt",
            None,
        )

    if not options:
        raise FileNotFoundError("No final inference or LOGO checkpoints were found.")
    if not default_label:
        default_label = next(iter(options))
    return options, default_label


@st.cache_resource
def load_model(checkpoint_path: str):
    device = get_device()
    model = build_model(num_classes=2).to(device)
    state = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


@st.cache_data
def get_image_size() -> int:
    if get_training_settings is None:
        return 224
    _, _, image_size, _ = get_training_settings()
    return int(image_size)


def predict(image: Image.Image, model, device, image_size: int):
    transform = get_eval_transform(image_size)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    prob_real = float(probs[0])
    prob_ai = float(probs[1])
    return prob_real, prob_ai


st.set_page_config(
    page_title="RealVision Demo",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
    :root {
        --bg: #f6f5f2;
        --card: #ffffff;
        --text: #0b1f2a;
        --muted: #5a6b75;
        --accent: #0b5b6e;
        --border: #e3e0da;
    }
    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }
    .block-container {
        padding-top: 2rem;
        max-width: 760px;
    }
    .rv-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 18px 10px 18px;
        box-shadow: 0 6px 22px rgba(0,0,0,0.06);
    }
    .rv-title {
        font-size: 28px;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .rv-subtitle {
        color: var(--muted);
        margin-bottom: 1rem;
    }
    .rv-label {
        color: var(--muted);
        font-size: 13px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .rv-pred {
        font-size: 22px;
        font-weight: 600;
        color: var(--accent);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="rv-title">RealVision</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="rv-subtitle">Upload an image to score it with the finalized deployment model or a LOGO benchmark checkpoint.</div>',
    unsafe_allow_html=True,
)

try:
    model_options, default_label = get_model_options()
except FileNotFoundError:
    st.error(
        "No final inference or LOGO checkpoints found. Train `final_inference` or `logo_eval` first."
    )
    st.stop()

option_labels = list(model_options.keys())
default_index = option_labels.index(default_label)

with st.sidebar:
    st.markdown("### Settings")
    selected_label = st.selectbox("Model", option_labels, index=default_index)
    selected_checkpoint, saved_threshold = model_options[selected_label]
    slider_default = float(saved_threshold) if saved_threshold is not None else 0.50
    manual_threshold = st.slider(
        "Decision threshold (prob_ai)",
        min_value=0.0,
        max_value=1.0,
        value=max(0.0, min(1.0, slider_default)),
        step=0.01,
    )


uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    if not selected_checkpoint.exists():
        st.error(f"Checkpoint not found: {selected_checkpoint}")
        st.stop()

    decision_threshold = float(manual_threshold)

    model, device = load_model(str(selected_checkpoint))
    image_size = get_image_size()
    prob_real, prob_ai = predict(image, model, device, image_size)

    pred_label = "ai_generated" if prob_ai >= decision_threshold else "real"

    with st.container():
        st.markdown('<div class="rv-card">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded image", use_container_width=True)
        st.markdown('<div class="rv-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rv-pred">{pred_label}</div>', unsafe_allow_html=True)
        st.write(f"Prob AI: **{prob_ai:.4f}**")
        st.write(f"Prob Real: **{prob_real:.4f}**")
        st.write(f"Model: **{selected_label}**")
        st.write(f"Decision threshold: **{decision_threshold:.2f}**")
        st.caption("Decision rule: predict AI if prob_ai ≥ threshold, else real.")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown('<div class="rv-card">', unsafe_allow_html=True)
    st.write("Drop an image here to get a prediction.")
    st.markdown("</div>", unsafe_allow_html=True)
