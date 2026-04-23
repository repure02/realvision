import argparse
import csv
import hashlib
import io
import random
import time
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from src.utils.config import get_paths_config, load_dataset_config

CONFIG = load_dataset_config()
PATHS_CONFIG = get_paths_config(CONFIG)

DEFAULT_SAVE_DIR = Path("data/raw/ai_generated/rapidata_non_sd")
META_DIR = PATHS_CONFIG.get("metadata_dir") or Path("data/metadata")
META_CSV = META_DIR / "rapidata_non_sd_metadata.csv"

raw_ai_dir = PATHS_CONFIG.get("raw_ai_dir")
SAVE_DIR = (raw_ai_dir / "rapidata_non_sd") if raw_ai_dir else DEFAULT_SAVE_DIR
SAVE_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TARGET_PER_GENERATOR = 300
DEFAULT_SEED = 42
DEFAULT_SPLIT = "train"


GENERATOR_SPECS = {
    "ideogram_v2": {
        "dataset_id": "Rapidata/Ideogram-V2_t2i_human_preference",
        "generator_family": "ideogram",
    },
    "recraft_v2": {
        "dataset_id": "Rapidata/Recraft-V2_t2i_human_preference",
        "generator_family": "recraft",
    },
    "imagen4": {
        "dataset_id": "Rapidata/Imagen4_t2i_human_preference",
        "generator_family": "imagen",
    },
    "openai_4o": {
        "dataset_id": "Rapidata/OpenAI-4o_t2i_human_preference",
        "generator_family": "openai",
    },
    "hidream_i1": {
        "dataset_id": "Rapidata/Hidream_t2i_human_preference",
        "generator_family": "hidream",
    },
}

GENERATOR_ALIASES = {
    "ideogram2": "ideogram_v2",
    "recraft2": "recraft_v2",
    "gpt4o": "openai_4o",
    "hidream": "hidream_i1",
}


def ensure_csv() -> None:
    if META_CSV.exists():
        return

    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "filepath",
            "label",
            "source",
            "generator_name",
            "generator_family",
            "width",
            "height",
            "format",
            "collection_date",
            "split",
            "notes",
            "caption",
        ])


def load_existing_ids() -> set[str]:
    existing_ids: set[str] = set()

    if not META_CSV.exists():
        return existing_ids

    with open(META_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id")
            if image_id:
                existing_ids.add(image_id)

    return existing_ids


def infer_target_columns(dataset) -> tuple[str, str]:
    model1_values = set(dataset.unique("model1")) if "model1" in dataset.column_names else set()
    model2_values = set(dataset.unique("model2")) if "model2" in dataset.column_names else set()

    if len(model1_values) == 1 and len(model2_values) > 1:
        return "model1", "image1"

    if len(model2_values) == 1 and len(model1_values) > 1:
        return "model2", "image2"

    raise ValueError(
        "Could not infer which image column belongs to the target generator. "
        f"model1 unique={len(model1_values)}, model2 unique={len(model2_values)}"
    )


def normalize_format(image: Image.Image) -> tuple[Image.Image, str]:
    image_format = (image.format or "PNG").upper()
    if image_format == "JPEG":
        ext = "jpg"
    elif image_format == "WEBP":
        ext = "webp"
    else:
        ext = "png"
        image_format = "PNG"

    if image_format in {"JPEG", "WEBP"} and image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")

    return image, ext


def image_bytes_and_size(image: Image.Image, ext: str) -> tuple[bytes, tuple[int, int]]:
    width, height = image.size
    buffer = io.BytesIO()

    save_format = {
        "jpg": "JPEG",
        "png": "PNG",
        "webp": "WEBP",
    }[ext]

    if save_format == "JPEG" and image.mode != "RGB":
        image = image.convert("RGB")

    image.save(buffer, format=save_format)
    return buffer.getvalue(), (width, height)


def select_generators(generator_names: list[str] | None) -> list[str]:
    if not generator_names:
        return list(GENERATOR_SPECS.keys())

    normalized_names = [GENERATOR_ALIASES.get(name, name) for name in generator_names]
    invalid = [name for name in normalized_names if name not in GENERATOR_SPECS]
    if invalid:
        raise ValueError(
            f"Unknown generators requested: {invalid}. "
            f"Valid values: {sorted(GENERATOR_SPECS)}. "
            f"Accepted aliases: {sorted(GENERATOR_ALIASES)}"
        )

    seen = set()
    ordered = []
    for name in normalized_names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def collect_generator(
    generator_name: str,
    target_total: int,
    split: str,
    rng: random.Random,
    existing_ids: set[str],
) -> int:
    spec = GENERATOR_SPECS[generator_name]
    dataset_id = spec["dataset_id"]
    generator_family = spec["generator_family"]

    print(f"\n[INFO] Loading {dataset_id}")
    dataset = load_dataset(dataset_id, split=split)
    model_col, image_col = infer_target_columns(dataset)
    target_model_name = dataset[0][model_col]

    print(
        f"[INFO] Collecting generator='{generator_name}' from {image_col} "
        f"(dataset model name: {target_model_name})"
    )

    save_dir = SAVE_DIR / generator_name
    save_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    saved = 0
    seen_hashes: set[str] = set()

    with open(META_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for idx in indices:
            if saved >= target_total:
                break

            row = dataset[idx]
            image = row[image_col]
            if image is None:
                continue

            image, ext = normalize_format(image)
            image_bytes, (width, height) = image_bytes_and_size(image, ext)
            content_hash = hashlib.sha1(image_bytes).hexdigest()

            if content_hash in seen_hashes:
                continue

            image_id = f"{generator_name}_{content_hash[:16]}"
            if image_id in existing_ids:
                seen_hashes.add(content_hash)
                continue

            filename = f"{image_id}.{ext}"
            save_path = save_dir / filename

            with open(save_path, "wb") as img_file:
                img_file.write(image_bytes)

            prompt = str(row.get("prompt", "") or "")
            notes = (
                f"dataset_id={dataset_id}; split={split}; source_model={target_model_name}; "
                f"row_index={idx}"
            )

            writer.writerow([
                image_id,
                str(save_path).replace("\\", "/"),
                "ai_generated",
                "rapidata_hf",
                generator_name,
                generator_family,
                width,
                height,
                ext,
                time.strftime("%Y-%m-%d"),
                "",
                notes,
                prompt,
            ])
            f.flush()

            existing_ids.add(image_id)
            seen_hashes.add(content_hash)
            saved += 1

            if saved % 25 == 0 or saved == target_total:
                print(f"[OK] {generator_name}: saved {saved}/{target_total}")

    print(f"[DONE] {generator_name}: saved {saved} unique images.")
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect non-Stable-Diffusion AI images from free Rapidata Hugging Face datasets."
    )
    parser.add_argument(
        "--generators",
        nargs="*",
        default=None,
        help=(
            "Subset of generators to collect. "
            f"Canonical values: {', '.join(sorted(GENERATOR_SPECS))}. "
            f"Accepted aliases: {', '.join(sorted(GENERATOR_ALIASES))}"
        ),
    )
    parser.add_argument(
        "--target_per_generator",
        type=int,
        default=DEFAULT_TARGET_PER_GENERATOR,
        help="How many unique images to save per generator.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Dataset split to load from Hugging Face.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for row shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generators = select_generators(args.generators)

    ensure_csv()
    existing_ids = load_existing_ids()
    rng = random.Random(args.seed)

    print(f"[INFO] Output directory: {SAVE_DIR}")
    print(f"[INFO] Metadata CSV: {META_CSV}")
    print(f"[INFO] Generators: {generators}")
    print(f"[INFO] Target per generator: {args.target_per_generator}")

    total_saved = 0
    for generator_name in generators:
        saved = collect_generator(
            generator_name=generator_name,
            target_total=args.target_per_generator,
            split=args.split,
            rng=rng,
            existing_ids=existing_ids,
        )
        total_saved += saved

    print(f"\n[DONE] Total newly saved images: {total_saved}")


if __name__ == "__main__":
    main()
