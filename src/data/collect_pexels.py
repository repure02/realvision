import argparse
import time
import csv
import requests
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

from src.utils.config import get_env_or_config, load_dataset_config, get_paths_config
load_dotenv()

CONFIG = load_dataset_config()
PATHS_CONFIG = get_paths_config(CONFIG)
API_KEY = get_env_or_config("PEXELS_API_KEY", CONFIG)
BASE_URL = "https://api.pexels.com/v1/search"

DEFAULT_SAVE_DIR = Path("data/raw/real/pexels")
META_DIR = PATHS_CONFIG.get("metadata_dir") or Path("data/metadata")
META_CSV = META_DIR / "pexels_metadata.csv"

raw_real_dir = PATHS_CONFIG.get("raw_real_dir")
SAVE_DIR = (raw_real_dir / "pexels") if raw_real_dir else DEFAULT_SAVE_DIR
SAVE_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"Authorization": API_KEY}

QUERIES = [
    "people",
    "street",
    "city",
    "nature",
    "indoor",
    "food",
    "travel",
    "animals",
    "sports",
    "architecture",
]

TARGET_TOTAL = 1000
PER_PAGE = 80
MIN_WIDTH = 512
MIN_HEIGHT = 512
SLEEP_BETWEEN_REQUESTS = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Collect real photographs from Pexels.")
    parser.add_argument(
        "--target_total",
        type=int,
        default=TARGET_TOTAL,
        help="How many new images to download in this run.",
    )
    return parser.parse_args()


def get_extension_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".png"):
        return ".png"
    return ".jpg"


def search_photos(query: str, page: int = 1, per_page: int = 80):
    params = {
        "query": query,
        "page": page,
        "per_page": per_page,
    }
    response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    return response.json(), response.headers


def download_image(url: str, save_path: Path):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)


def main():
    args = parse_args()

    if not API_KEY:
        raise ValueError("PEXELS_API_KEY not found in environment.")

    existing_ids = set()
    if META_CSV.exists():
        with open(META_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(str(row["image_id"]))

    file_exists = META_CSV.exists()
    with open(META_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
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
                "pexels_url",
                "photographer",
                "query",
            ])

        saved = 0

        for query in QUERIES:
            page = 1
            while saved < args.target_total:
                try:
                    data, headers = search_photos(query=query, page=page, per_page=PER_PAGE)
                except Exception as e:
                    print(f"[WARN] Failed query={query}, page={page}: {e}")
                    break

                photos = data.get("photos", [])
                if not photos:
                    break

                remaining = headers.get("X-Ratelimit-Remaining", "unknown")
                print(f"[INFO] query={query}, page={page}, remaining={remaining}")

                for photo in photos:
                    if saved >= args.target_total:
                        break

                    image_id = str(photo["id"])
                    if image_id in existing_ids:
                        continue

                    width = photo.get("width", 0)
                    height = photo.get("height", 0)
                    if width < MIN_WIDTH or height < MIN_HEIGHT:
                        continue

                    image_url = photo["src"]["original"]
                    ext = get_extension_from_url(image_url)
                    save_path = SAVE_DIR / f"pexels_{image_id}{ext}"

                    try:
                        download_image(image_url, save_path)
                    except Exception as e:
                        print(f"[WARN] Failed download {image_id}: {e}")
                        continue

                    writer.writerow([
                        image_id,
                        str(save_path).replace("\\", "/"),
                        "real",
                        "pexels",
                        "",
                        "",
                        width,
                        height,
                        ext.replace(".", ""),
                        time.strftime("%Y-%m-%d"),
                        "",
                        "",
                        photo.get("url", ""),
                        photo.get("photographer", ""),
                        query,
                    ])
                    f.flush()

                    existing_ids.add(image_id)
                    saved += 1
                    print(f"[OK] Saved {saved}: {save_path.name}")

                page += 1
                time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"[DONE] Downloaded {saved} new images.")


if __name__ == "__main__":
    main()
