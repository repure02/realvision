from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, args: list[str]) -> None:
    print(f"\n=== {name} ===")
    subprocess.run(args, check=True, cwd=PROJECT_ROOT)


def run_module_step(name: str, module: str, extra_args: list[str] | None = None) -> None:
    args = [sys.executable, "-m", module]
    if extra_args:
        args.extend(extra_args)
    run_step(name, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RealVision pipeline.")
    parser.add_argument(
        "--start_from",
        type=str,
        default="collection",
        choices=["collection", "master_metadata", "train"],
        help="Where to start in the pipeline.",
    )
    parser.add_argument(
        "--split_column",
        type=str,
        default="heldout_generator_split",
        choices=["random_split", "heldout_generator_split"],
        help="Which split to use for training/evaluation.",
    )
    parser.add_argument("--run_both_splits", action="store_true", help="Run both split types for train/eval.")
    parser.add_argument("--skip_pexels", action="store_true", help="Skip Pexels collection.")
    parser.add_argument("--skip_wikimedia", action="store_true", help="Skip Wikimedia collection.")
    parser.add_argument("--skip_defactify", action="store_true", help="Skip Defactify collection.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    args = parser.parse_args()

    python = sys.executable

    if args.start_from == "collection":
        if not args.skip_pexels:
            run_module_step("Collect Pexels", "src.data.collect_pexels")
        if not args.skip_wikimedia:
            run_module_step("Collect Wikimedia", "src.data.collect_wikimedia")
        if not args.skip_defactify:
            run_module_step("Collect Defactify", "src.data.collect_defactify")

    if args.start_from in {"collection", "master_metadata"}:
        run_module_step("Build Master Metadata", "src.data.build_master_metadata")

    if args.start_from in {"collection", "master_metadata"}:
        run_module_step("Process Images", "src.data.process_images")

    if args.start_from in {"collection", "master_metadata"}:
        run_module_step("Create Splits", "src.data.create_splits")

    if args.start_from in {"collection", "master_metadata", "train"}:
        split_columns = (
            ["random_split", "heldout_generator_split"]
            if args.run_both_splits
            else [args.split_column]
        )
        for split_column in split_columns:
            run_module_step(
                f"Train ({split_column})",
                "src.training.train",
                ["--split_column", split_column, "--epochs", str(args.epochs)],
            )

    if args.start_from in {"collection", "master_metadata", "train"}:
        split_columns = (
            ["random_split", "heldout_generator_split"]
            if args.run_both_splits
            else [args.split_column]
        )
        for split_column in split_columns:
            run_module_step(
                f"Evaluate ({split_column})",
                "src.training.evaluate",
                ["--split_column", split_column, "--run_tag", split_column],
            )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
