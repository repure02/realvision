from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_threshold(path: Path) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("threshold="):
            return float(line.split("=", 1)[1])
    raise ValueError(f"No threshold= line found in {path}")


def metric_row(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def assert_close(name: str, actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{name} mismatch: actual={actual}, expected={expected}")


def validate_threshold_outputs(
    metrics_path: Path,
    sweep_path: Path,
    threshold_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    missing = [
        path
        for path in (metrics_path, sweep_path, threshold_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing final artifacts: " + ", ".join(str(path) for path in missing))

    threshold = load_threshold(threshold_path)
    sweep_df = pd.read_csv(sweep_path)
    required_sweep_columns = {"threshold", "accuracy", "precision", "recall", "f1"}
    missing_sweep_columns = required_sweep_columns - set(sweep_df.columns)
    if missing_sweep_columns:
        raise ValueError(f"Missing threshold sweep columns: {sorted(missing_sweep_columns)}")
    if list(sweep_df["threshold"]) != sorted(sweep_df["threshold"]):
        raise ValueError("Threshold sweep must be sorted by threshold ascending.")

    chosen_rows = sweep_df[sweep_df["threshold"].round(6) == round(threshold, 6)]
    if chosen_rows.empty:
        raise ValueError(f"Chosen threshold {threshold:.2f} is missing from threshold sweep.")

    metrics_df = pd.read_csv(metrics_path)
    required_metric_columns = {
        "generator_name",
        "count",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "accuracy",
    }
    missing_metric_columns = required_metric_columns - set(metrics_df.columns)
    if missing_metric_columns:
        raise ValueError(f"Missing generator metric columns: {sorted(missing_metric_columns)}")

    return metrics_df, sweep_df, threshold


def validate_final_artifacts(
    predictions_path: Path,
    metrics_path: Path,
    sweep_path: Path,
    threshold_path: Path,
    allow_missing_predictions: bool = False,
) -> None:
    metrics_df, sweep_df, threshold = validate_threshold_outputs(
        metrics_path=metrics_path,
        sweep_path=sweep_path,
        threshold_path=threshold_path,
    )

    if not predictions_path.exists():
        if allow_missing_predictions:
            print(f"Validated published final artifacts at threshold={threshold:.2f}")
            print("Calibration predictions CSV is not included in the published bundle.")
            print(f"Generator rows: {len(metrics_df)}")
            return
        raise FileNotFoundError(f"Missing final artifacts: {predictions_path}")

    predictions_df = pd.read_csv(predictions_path)
    required_prediction_columns = {
        "image_id",
        "generator_name",
        "true_label",
        "pred_label",
        "prob_ai",
    }
    missing_columns = required_prediction_columns - set(predictions_df.columns)
    if missing_columns:
        raise ValueError(f"Missing prediction columns: {sorted(missing_columns)}")

    expected_pred = (predictions_df["prob_ai"].astype(float) >= threshold).astype(int)
    actual_pred = predictions_df["pred_label"].astype(int)
    mismatches = int((expected_pred != actual_pred).sum())
    if mismatches:
        raise ValueError(
            f"pred_label is inconsistent with threshold={threshold:.2f}; mismatches={mismatches}"
        )

    if "decision_threshold" in predictions_df.columns:
        thresholds = set(round(float(value), 6) for value in predictions_df["decision_threshold"].dropna())
        if thresholds != {round(threshold, 6)}:
            raise ValueError(f"decision_threshold column does not match chosen threshold: {thresholds}")

    chosen_rows = sweep_df[sweep_df["threshold"].round(6) == round(threshold, 6)]
    computed = metric_row(predictions_df["true_label"].astype(int), actual_pred)
    chosen = chosen_rows.iloc[0]
    for metric_name, value in computed.items():
        assert_close(f"sweep {metric_name}", float(chosen[metric_name]), value)

    expected_rows = []
    for generator, group in predictions_df.groupby("generator_name", sort=True):
        y_true = group["true_label"].astype(int)
        y_pred = group["pred_label"].astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        expected_rows.append(
            {
                "generator_name": generator,
                "count": int(len(group)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                **metric_row(y_true, y_pred),
            }
        )

    expected_df = pd.DataFrame(expected_rows).sort_values("generator_name").reset_index(drop=True)
    actual_df = metrics_df.sort_values("generator_name").reset_index(drop=True)
    if list(expected_df["generator_name"]) != list(actual_df["generator_name"]):
        raise ValueError("Generator metric rows do not match prediction generators.")

    for column in ("count", "tp", "fp", "fn", "tn"):
        if not (expected_df[column].astype(int) == actual_df[column].astype(int)).all():
            raise ValueError(f"Generator metric column mismatch: {column}")
    for column in ("accuracy", "precision", "recall", "f1"):
        for idx, expected in enumerate(expected_df[column]):
            assert_close(
                f"{actual_df.loc[idx, 'generator_name']} {column}",
                float(actual_df.loc[idx, column]),
                round(float(expected), 6),
            )

    print(f"Validated final artifacts at threshold={threshold:.2f}")
    print(f"Rows: {len(predictions_df)}")
    print(
        "Metrics: "
        + ", ".join(f"{name}={value:.4f}" for name, value in computed.items())
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final inference reports and threshold contract.")
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="reports/final_inference_calibration_predictions.csv",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="reports/final_inference_generator_metrics.csv",
    )
    parser.add_argument(
        "--sweep_path",
        type=str,
        default="reports/final_inference_threshold_sweep.csv",
    )
    parser.add_argument(
        "--threshold_path",
        type=str,
        default="reports/final_inference_chosen_threshold.txt",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Exit successfully if final artifacts are not present, useful for lightweight CI clones.",
    )
    parser.add_argument(
        "--allow_missing_predictions",
        action="store_true",
        help="Allow the calibration predictions CSV to be absent, useful for validating the published bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        resolve_path(args.predictions_path),
        resolve_path(args.metrics_path),
        resolve_path(args.sweep_path),
        resolve_path(args.threshold_path),
    ]
    if args.allow_missing and any(not path.exists() for path in paths):
        print("Final inference artifacts not found; skipping final artifact validation.")
        return
    validate_final_artifacts(*paths, allow_missing_predictions=args.allow_missing_predictions)


if __name__ == "__main__":
    main()
