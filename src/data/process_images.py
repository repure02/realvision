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

    processed_rows = []

    max_side = PROCESSING_CONFIG["max_side"]
    jpeg_quality = PROCESSING_CONFIG["jpeg_quality"]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_path = PROJECT_ROOT / row["filepath"]

        output_filename = Path(row["filename"]).stem + ".jpg"
        output_path = OUTPUT_DIR / row["label"] / output_filename

        success = process_image(input_path, output_path, max_side, jpeg_quality)

        if success:
            new_row = row.copy()
            new_row["filepath"] = str(output_path.relative_to(PROJECT_ROOT))
            new_row["format"] = "jpg"
            new_row["width"], new_row["height"] = Image.open(output_path).size
            processed_rows.append(new_row)

    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(OUTPUT_METADATA, index=False)

    print(f"\nSaved processed metadata to: {OUTPUT_METADATA}")
    print(f"Total processed images: {len(processed_df)}")


if __name__ == "__main__":
    main()
