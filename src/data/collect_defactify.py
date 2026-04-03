import csv
import random
import time
from pathlib import Path

from datasets import load_dataset

from src.utils.config import get_paths_config, load_dataset_config

CONFIG = load_dataset_config()
PATHS_CONFIG = get_paths_config(CONFIG)

DEFAULT_SAVE_DIR = Path("data/raw/ai_generated/defactify")
META_DIR = PATHS_CONFIG.get("metadata_dir") or Path("data/metadata")
META_CSV = META_DIR / "defactify_metadata.csv"

raw_ai_dir = PATHS_CONFIG.get("raw_ai_dir")
SAVE_DIR = (raw_ai_dir / "defactify") if raw_ai_dir else DEFAULT_SAVE_DIR
SAVE_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TOTAL = 2000
SEED = 42

MODEL_MAP = {
    1: "sd21",
    2: "sdxl",
    3: "sd3",
    4: "dalle3",
    5: "midjourney",
}


def ensure_csv():
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
            "label_a",
            "label_b",
        ])


def load_existing_ids():
    existing = set()
    if META_CSV.exists():
        with open(META_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["image_id"])
    return existing


def main():
    random.seed(SEED)
    ensure_csv()
    existing_ids = load_existing_ids()

    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")

    saved = 0

    with open(META_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for split_name in ["train", "validation", "test"]:
            split_ds = dataset[split_name]

            indices = list(range(len(split_ds)))
            random.shuffle(indices)

            for idx in indices:
                if saved >= TARGET_TOTAL:
                    break

                row = split_ds[idx]

                if int(row["Label_A"]) != 1:
                    continue

                image_id = f"defactify_{split_name}_{idx}"
                if image_id in existing_ids:
                    continue

                model_id = int(row["Label_B"])
                generator_name = MODEL_MAP.get(model_id, "unknown")

                image = row["Image"]
                width, height = image.size

                ext = "png"
                filename = f"{image_id}.{ext}"
                save_path = SAVE_DIR / filename

                image.save(save_path)

                writer.writerow([
                    image_id,
                    str(save_path).replace("\\", "/"),
                    "ai_generated",
                    "defactify",
                    generator_name,
                    "diffusion" if generator_name != "midjourney" else "proprietary",
                    width,
                    height,
                    ext,
                    time.strftime("%Y-%m-%d"),
                    "",
                    "",
                    row.get("Caption", ""),
                    row["Label_A"],
                    row["Label_B"],
                ])

                existing_ids.add(image_id)
                saved += 1
                print(f"[OK] Saved {saved}: {filename}")

            if saved >= TARGET_TOTAL:
                break

    print(f"[DONE] Downloaded {saved} AI-generated images from Defactify.")


if __name__ == "__main__":
    main()
