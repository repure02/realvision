from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE_CHOICES = ["data_prep", "train", "logo_eval", "final_inference", "reports", "full"]


def run_step(name: str, args: list[str]) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(args))
    subprocess.run(args, check=True, cwd=PROJECT_ROOT)


def run_module_step(name: str, module: str, extra_args: list[str] | None = None) -> None:
    args = [sys.executable, "-m", module]
    if extra_args:
        args.extend(extra_args)
    run_step(name, args)


def run_collection(args: argparse.Namespace) -> None:
    if not args.skip_pexels:
        run_module_step("Collect Pexels", "src.data.collect_pexels")
    if not args.skip_wikimedia:
        run_module_step("Collect Wikimedia", "src.data.collect_wikimedia")
    if not args.skip_defactify:
        run_module_step("Collect Defactify", "src.data.collect_defactify")
    if not args.skip_rapidata_non_sd:
        run_module_step("Collect Rapidata Non-SD", "src.data.collect_rapidata_non_sd")


def run_data_prep(args: argparse.Namespace) -> None:
    if args.include_collection:
        run_collection(args)

    run_module_step("Build Master Metadata", "src.data.build_master_metadata")
    run_module_step("Process Images", "src.data.process_images")
    run_module_step("Create Splits", "src.data.create_splits")


def run_logo_stage(args: argparse.Namespace) -> None:
    logo_args = ["--epochs", str(args.logo_epochs or args.epochs), "--loss", args.loss]
    if args.logo_test_generator:
        logo_args.extend(["--logo_test_generator", args.logo_test_generator])
    else:
        logo_args.append("--logo_all")
    if args.focal_gamma is not None:
        logo_args.extend(["--focal_gamma", str(args.focal_gamma)])
    if args.logo_val_generator:
        logo_args.extend(["--logo_val_generator", args.logo_val_generator])

    name = (
        f"Train LOGO Run ({args.logo_test_generator})"
        if args.logo_test_generator
        else "Train LOGO Runs"
    )
    run_module_step(name, "src.training.train", logo_args)


def run_final_inference_stage(args: argparse.Namespace) -> None:
    final_args = [
        "--final_inference",
        "--epochs",
        str(args.final_epochs or args.epochs),
        "--loss",
        args.loss,
        "--final_val_fraction",
        str(args.final_val_fraction),
        "--target_recall",
        str(args.target_recall),
    ]
    if args.focal_gamma is not None:
        final_args.extend(["--focal_gamma", str(args.focal_gamma)])
    run_module_step("Train Final Inference Model", "src.training.train", final_args)


def run_reports_stage(args: argparse.Namespace) -> None:
    run_module_step("Generate Dataset Specs", "src.utils.generate_dataset_specs")

    if args.backfill_logo_details:
        backfill_args: list[str] = []
        if args.backfill_missing_only:
            backfill_args.append("--missing_only")
        run_module_step("Backfill LOGO Details", "src.utils.backfill_logo_details", backfill_args)

    run_module_step("Generate LOGO Failure Analysis", "src.utils.generate_logo_failure_analysis")
    run_module_step("Generate LOGO Report", "src.utils.generate_logo_report")
    run_module_step("Validate LOGO Baseline", "src.utils.validate_logo_baseline")
    run_module_step("Generate Baseline Manifest", "src.utils.generate_baseline_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RealVision pipeline in reproducible stages. "
            "The training workflow uses LOGO for benchmarking and a dedicated final-inference "
            "stage for deployment, writing canonical outputs under "
            "data/, checkpoints/, reports/, and runs/. "
            "Use `final_inference` to train the single deployment checkpoint."
        )
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=STAGE_CHOICES,
        help="Pipeline stage to run. `train` is kept as an alias for `logo_eval`.",
    )
    parser.add_argument(
        "--include_collection",
        action="store_true",
        help="Run data collection before metadata/image processing in the data_prep stage.",
    )
    parser.add_argument("--skip_pexels", action="store_true", help="Skip Pexels collection.")
    parser.add_argument("--skip_wikimedia", action="store_true", help="Skip Wikimedia collection.")
    parser.add_argument("--skip_defactify", action="store_true", help="Skip Defactify collection.")
    parser.add_argument(
        "--skip_rapidata_non_sd",
        action="store_true",
        help="Skip free non-SD Hugging Face Rapidata collection.",
    )
    parser.add_argument(
        "--logo_test_generator",
        type=str,
        default=None,
        help="Optional single LOGO test generator to train instead of the full generator sweep.",
    )
    parser.add_argument(
        "--logo_val_generator",
        type=str,
        default=None,
        help="Optional validation generator override for every LOGO run.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Default epoch count for training stages.")
    parser.add_argument(
        "--logo_epochs",
        type=int,
        default=None,
        help="Optional epoch override for LOGO runs. Defaults to --epochs.",
    )
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=None,
        help="Optional epoch override for the final inference model. Defaults to --epochs.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Loss function passed through to training.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma when --loss=focal.",
    )
    parser.add_argument(
        "--final_val_fraction",
        type=float,
        default=0.15,
        help="Validation fraction carved from the random train/val pool for final inference training.",
    )
    parser.add_argument(
        "--target_recall",
        type=float,
        default=0.8,
        help="Target recall used when choosing the saved threshold for the final inference checkpoint.",
    )
    parser.add_argument(
        "--backfill_logo_details",
        action="store_true",
        help="Rebuild LOGO detail CSVs from existing checkpoints before generating reports.",
    )
    parser.add_argument(
        "--backfill_missing_only",
        action="store_true",
        help="Only backfill missing LOGO detail files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in {"data_prep", "full"}:
        run_data_prep(args)

    if args.stage in {"train", "logo_eval", "full"}:
        run_logo_stage(args)

    if args.stage in {"final_inference", "full"}:
        run_final_inference_stage(args)

    if args.stage in {"reports", "full"}:
        run_reports_stage(args)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
