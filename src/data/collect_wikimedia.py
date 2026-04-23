import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

from src.utils.config import get_paths_config, load_dataset_config

API_URL = "https://commons.wikimedia.org/w/api.php"

CONFIG = load_dataset_config()
PATHS_CONFIG = get_paths_config(CONFIG)

DEFAULT_SAVE_DIR = Path("data/raw/real/wikimedia")
META_DIR = PATHS_CONFIG.get("metadata_dir") or Path("data/metadata")
META_CSV = META_DIR / "wikimedia_metadata.csv"

raw_real_dir = PATHS_CONFIG.get("raw_real_dir")
SAVE_DIR = (raw_real_dir / "wikimedia") if raw_real_dir else DEFAULT_SAVE_DIR

SAVE_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# Target how many NEW images to download in this run
TARGET_TOTAL = 1000

# Keep requests gentle
SLEEP_BETWEEN_REQUESTS = 0.8
TIMEOUT = 30

# Basic quality filter
MIN_WIDTH = 512
MIN_HEIGHT = 512

# These categories are much safer than broad commons search.
# They are still not perfect, but they greatly reduce drawings/logos/etc.
CATEGORIES = [
    "Category:Photographs by genre",
    "Category:Photographs by subject",
    "Category:Featured pictures on Wikimedia Commons",
]

# Optional: skip file extensions that are often not photographs
BAD_EXTENSIONS = {".svg", ".gif", ".tif", ".tiff", ".webm", ".ogg", ".ogv", ".djvu", ".pdf"}


session = requests.Session()
session.headers.update(
    {
        "User-Agent": "RealVisionDatasetCollector/1.0 (educational portfolio project; contact: razvan.lemond@gmail.com)"
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect real photographs from Wikimedia Commons.")
    parser.add_argument(
        "--target_total",
        type=int,
        default=TARGET_TOTAL,
        help="How many new images to download in this run.",
    )
    return parser.parse_args()


def api_get(params: Dict) -> Dict:
    params = dict(params)
    params["format"] = "json"
    response = session.get(API_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def load_existing_ids_and_titles() -> Tuple[Set[str], Set[str]]:
    existing_ids: Set[str] = set()
    existing_titles: Set[str] = set()

    if META_CSV.exists():
        with open(META_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("image_id"):
                    existing_ids.add(row["image_id"])
                if row.get("page_title"):
                    existing_titles.add(row["page_title"])

    return existing_ids, existing_titles


def ensure_csv_header() -> None:
    if META_CSV.exists():
        return

    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
                "page_title",
                "description_url",
                "uploader",
                "license_short_name",
                "category_seed",
            ]
        )


def get_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix


def list_category_files(category_title: str, cmcontinue: Optional[str] = None, cmlimit: int = 100) -> Tuple[List[Dict], Optional[str]]:
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtype": "file",
        "cmtitle": category_title,
        "cmlimit": cmlimit,
    }
    if cmcontinue:
        params["cmcontinue"] = cmcontinue

    data = api_get(params)
    members = data.get("query", {}).get("categorymembers", [])
    next_token = data.get("continue", {}).get("cmcontinue")
    return members, next_token


def fetch_imageinfo_for_title(title: str) -> Optional[Dict]:
    params = {
        "action": "query",
        "prop": "imageinfo",
        "titles": title,
        "iiprop": "url|size|user|extmetadata",
    }

    data = api_get(params)
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    imageinfo_list = page.get("imageinfo")
    if not imageinfo_list:
        return None

    info = imageinfo_list[0]
    info["_pageid"] = str(page.get("pageid", ""))
    info["_title"] = page.get("title", title)
    return info


def looks_like_photo_candidate(title: str, info: Dict) -> bool:
    ext = get_extension(title)
    if ext in BAD_EXTENSIONS:
        return False

    width = int(info.get("width", 0) or 0)
    height = int(info.get("height", 0) or 0)
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False

    url = info.get("url", "")
    if not url:
        return False

    extmeta = info.get("extmetadata", {}) or {}

    # Commons metadata often exposes license and descriptions via extmetadata.
    # We use a few soft filters to avoid obvious non-photo items.
    image_description = (
        extmeta.get("ImageDescription", {}).get("value", "") if isinstance(extmeta.get("ImageDescription"), dict) else ""
    ).lower()
    object_name = (
        extmeta.get("ObjectName", {}).get("value", "") if isinstance(extmeta.get("ObjectName"), dict) else ""
    ).lower()

    suspicious_terms = [
        "illustration",
        "drawing",
        "logo",
        "diagram",
        "map",
        "coat of arms",
        "painting",
        "flag",
        "seal",
        "icon",
        "poster",
        "svg",
    ]
    joined_text = f"{image_description} {object_name}"

    if any(term in joined_text for term in suspicious_terms):
        return False

    return True


def download_file(url: str, save_path: Path) -> None:
    response = session.get(url, timeout=TIMEOUT, stream=True)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def append_metadata_row(row: List) -> None:
    with open(META_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def safe_extmetadata_value(extmetadata: Dict, key: str) -> str:
    value = extmetadata.get(key, "")
    if isinstance(value, dict):
        return value.get("value", "")
    return value if isinstance(value, str) else ""


def main() -> None:
    args = parse_args()
    ensure_csv_header()
    existing_ids, existing_titles = load_existing_ids_and_titles()

    saved = 0

    for category in CATEGORIES:
        print(f"[INFO] Scanning {category}")
        cmcontinue = None

        while saved < args.target_total:
            try:
                members, cmcontinue = list_category_files(category, cmcontinue=cmcontinue, cmlimit=100)
            except Exception as e:
                print(f"[WARN] Category fetch failed for {category}: {e}")
                break

            if not members:
                break

            for member in members:
                if saved >= args.target_total:
                    break

                title = member.get("title", "")
                if not title or title in existing_titles:
                    continue

                try:
                    info = fetch_imageinfo_for_title(title)
                except Exception as e:
                    print(f"[WARN] imageinfo failed for {title}: {e}")
                    continue

                if not info:
                    continue

                pageid = info.get("_pageid", "")
                if pageid and pageid in existing_ids:
                    continue

                if not looks_like_photo_candidate(title, info):
                    continue

                ext = get_extension(title)
                filename = f"wikimedia_{pageid or title.replace('File:', '').replace(' ', '_')}{ext}"
                save_path = SAVE_DIR / filename

                try:
                    download_file(info["url"], save_path)
                except Exception as e:
                    print(f"[WARN] Download failed for {title}: {e}")
                    continue

                extmetadata = info.get("extmetadata", {}) or {}

                row = [
                    pageid,
                    str(save_path).replace("\\", "/"),
                    "real",
                    "wikimedia_commons",
                    "",
                    "",
                    int(info.get("width", 0) or 0),
                    int(info.get("height", 0) or 0),
                    ext.replace(".", "").lower(),
                    time.strftime("%Y-%m-%d"),
                    "",
                    "",
                    title,
                    info.get("descriptionurl", ""),
                    info.get("user", ""),
                    safe_extmetadata_value(extmetadata, "LicenseShortName"),
                    category,
                ]

                append_metadata_row(row)

                if pageid:
                    existing_ids.add(pageid)
                existing_titles.add(title)

                saved += 1
                print(f"[OK] Saved {saved}: {filename}")

                time.sleep(SLEEP_BETWEEN_REQUESTS)

            if not cmcontinue:
                break

    print(f"[DONE] Downloaded {saved} new Wikimedia Commons images.")


if __name__ == "__main__":
    main()
