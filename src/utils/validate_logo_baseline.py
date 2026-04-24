from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.logo_artifacts import PROJECT_ROOT, resolve_repo_path


REQUIRED_COLUMNS = {
    "best_val_recall",
    "test_loss",
    "test_acc",
    "test_recall",
    "test_generator",
    "val_generator",
    "generator_metrics_csv",
    "test_predictions_csv",
}


def validate_summary(summary_path: Path, allow_missing_details: bool = False) -> None:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {summary_path.name}: {sorted(missing_cols)}")

    if df["test_generator"].duplicated().any():
        dupes = sorted(df.loc[df["test_generator"].duplicated(), "test_generator"].astype(str).unique())
        raise ValueError(f"Duplicate test_generator values found: {dupes}")

    errors: list[str] = []
    missing_details: list[str] = []
    for _, row in df.iterrows():
        test_generator = str(row["test_generator"])
        for col in ("generator_metrics_csv", "test_predictions_csv"):
            value = str(row[col])
            if value.startswith("/content/") or value.startswith("realvision/"):
                errors.append(f"{test_generator}: stale path prefix in {col}: {value}")
                continue
            path = resolve_repo_path(value)
            if not path.exists():
                if allow_missing_details:
                    missing_details.append(f"{test_generator}: missing {col}: {value}")
                else:
                    errors.append(f"{test_generator}: missing {col}: {value}")
                continue

    if errors:
        raise ValueError("Baseline validation failed:\n- " + "\n- ".join(errors))

    avg_recall = pd.to_numeric(df["test_recall"], errors="coerce").mean()
    avg_acc = pd.to_numeric(df["test_acc"], errors="coerce").mean()

    print(f"Validated baseline: {summary_path}")
    print(f"Rows: {len(df)}")
    print(f"Average test recall: {avg_recall:.4f}")
    print(f"Average test accuracy: {avg_acc:.4f}")
    if missing_details:
        print(
            "Skipped missing detail artifact checks because --allow_missing_details was set:"
        )
        for item in missing_details:
            print(f"- {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the locked LOGO baseline artifacts.")
    parser.add_argument(
        "--summary_path",
        type=str,
        default="reports/logo_summary.csv",
        help="Path to the canonical LOGO summary CSV.",
    )
    parser.add_argument(
        "--allow_missing_details",
        action="store_true",
        help="Do not fail if referenced reports/details CSVs are absent, useful for lightweight CI clones.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    validate_summary(summary_path, allow_missing_details=args.allow_missing_details)


if __name__ == "__main__":
    main()
