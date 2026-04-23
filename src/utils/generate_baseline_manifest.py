from __future__ import annotations

import json
from datetime import date

import pandas as pd

from src.utils.logo_artifacts import PROJECT_ROOT, REPORTS_DIR


def main() -> None:
    summary_path = REPORTS_DIR / "logo_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing LOGO summary: {summary_path}")

    df = pd.read_csv(summary_path)
    if "test_generator" not in df.columns:
        raise ValueError("logo_summary.csv must include a test_generator column.")

    generators = sorted(df["test_generator"].dropna().astype(str).unique().tolist())
    manifest = {
        "baseline_name": f"expanded_{len(generators)}_generator_logo",
        "status": "official",
        "locked_on": date.today().isoformat(),
        "summary_csv": "reports/logo_summary.csv",
        "details_dir": "reports/details",
        "historical_comparison_csv": "reports/logo_summary_baseline_5gen.csv",
        "checkpoint_pattern": "checkpoints/convnext_tiny_logo_test_<generator>_best.pt",
        "generators": generators,
    }

    output_path = REPORTS_DIR / "baseline_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
