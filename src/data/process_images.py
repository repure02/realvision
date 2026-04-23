from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm

from src.utils.config import get_paths_config, get_processing_config, load_dataset_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG = load_dataset_config()
PATHS_CONFIG = get_paths_config(CONFIG)
PROCESSING_CONFIG = get_processing_config(CONFIG)

METADATA_DIR = PATHS_CONFIG.get("metadata_dir") or (PROJECT_ROOT / "data" / "metadata")
INPUT_METADATA = PATHS_CONFIG.get("master_metadata_csv") or (METADATA_DIR / "master_metadata.csv")
OUTPUT_METADATA = PATHS_CONFIG.get("processed_metadata_csv") or (METADATA_DIR / "processed_metadata.csv")

OUTPUT_DIR = PATHS_CONFIG.get("processed_images_dir") or (PROJECT_ROOT / "data" / "processed" / "images")


def load_existing_processed_rows() -> dict[str, dict]:
    if not OUTPUT_METADATA.exists():
        return {}

    existing_df = pd.read_csv(OUTPUT_METADATA)
    if "image_id" not in existing_df.columns:
        return {}

    existing_df["image_id"] = existing_df["image_id"].astype(str)
    return existing_df.set_index("image_id").to_dict("index")


def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def process_image(input_path: Path, output_path: Path, max_side: int, jpeg_quality: int):
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img = resize_keep_aspect(img, max_side)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format="JPEG", quality=jpeg_quality)

        return True
    except Exception as e:
        print(f"Failed: {input_path} -> {e}")
        return False


def main():
    df = pd.read_csv(INPUT_METADATA)
    df["image_id"] = df["image_id"].astype(str)

    processed_rows = []
    existing_rows = load_existing_processed_rows()
    reused_count = 0
    processed_count = 0

    max_side = PROCESSING_CONFIG["max_side"]
    jpeg_quality = PROCESSING_CONFIG["jpeg_quality"]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_path = PROJECT_ROOT / row["filepath"]

        output_filename = Path(row["filename"]).stem + ".jpg"
        output_path = OUTPUT_DIR / row["label"] / output_filename

        image_id = str(row["image_id"])
        existing_row = existing_rows.get(image_id)

        if existing_row and output_path.exists():
            reused_row = row.copy()
            for key, value in existing_row.items():
                reused_row[key] = value
            reused_row["filepath"] = str(output_path.relative_to(PROJECT_ROOT))
            processed_rows.append(reused_row)
            reused_count += 1
            continue

        success = process_image(input_path, output_path, max_side, jpeg_quality)

        if success:
            new_row = row.copy()
            new_row["filepath"] = str(output_path.relative_to(PROJECT_ROOT))
            new_row["format"] = "jpg"
            new_row["width"], new_row["height"] = Image.open(output_path).size
            processed_rows.append(new_row)
            processed_count += 1

    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(OUTPUT_METADATA, index=False)

    print(f"\nSaved processed metadata to: {OUTPUT_METADATA}")
    print(f"Total processed images: {len(processed_df)}")
    print(f"Reused existing processed images: {reused_count}")
    print(f"Newly processed images: {processed_count}")


if __name__ == "__main__":
    main()
