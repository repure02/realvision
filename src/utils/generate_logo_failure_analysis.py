from pathlib import Path

import pandas as pd

from src.utils.logo_artifacts import REPORTS_DIR, resolve_repo_path

FAILURES_DIR = REPORTS_DIR / "failures"


def _load_summary() -> pd.DataFrame:
    enriched = REPORTS_DIR / "logo_summary_enriched.csv"
    summary = REPORTS_DIR / "logo_summary.csv"
    if enriched.exists():
        return pd.read_csv(enriched)
    if summary.exists():
        return pd.read_csv(summary)
    raise FileNotFoundError(
        "Missing logo_summary.csv (or logo_summary_enriched.csv) in "
        f"{REPORTS_DIR}."
    )


def main(worst_k: int = 2) -> None:
    df = _load_summary()
    if "test_generator" not in df.columns or "test_recall" not in df.columns:
        raise ValueError("Summary must include test_generator and test_recall columns.")

    df["test_recall"] = pd.to_numeric(df["test_recall"], errors="coerce")
    worst = df.sort_values("test_recall", ascending=True).head(worst_k)

    FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for _, row in worst.iterrows():
        gen = str(row["test_generator"])
        pred_value = row.get("test_predictions_csv", f"reports/details/logo_test_{gen}_predictions.csv")
        pred_path = resolve_repo_path(str(pred_value))
        if not pred_path.exists():
            print(f"Missing predictions for {gen}: {pred_path}")
            continue

        preds = pd.read_csv(pred_path)
        required = {"true_label", "pred_label", "filepath", "generator_name"}
        if not required.issubset(set(preds.columns)):
            print(f"Skipping {gen}: missing columns in {pred_path.name}")
            continue

        fn = preds[(preds["true_label"] == 1) & (preds["pred_label"] == 0)].copy()
        fp = preds[(preds["true_label"] == 0) & (preds["pred_label"] == 1)].copy()

        fn_path = FAILURES_DIR / f"logo_{gen}_false_negatives.csv"
        fp_path = FAILURES_DIR / f"logo_{gen}_false_positives.csv"
        fn.to_csv(fn_path, index=False)
        fp.to_csv(fp_path, index=False)

        summary_rows.append(
            {
                "test_generator": gen,
                "test_recall": float(row["test_recall"]),
                "false_negatives": len(fn),
                "false_positives": len(fp),
                "fn_csv": str(fn_path),
                "fp_csv": str(fp_path),
            }
        )

        print(f"{gen}: saved {len(fn)} FN -> {fn_path.name}, {len(fp)} FP -> {fp_path.name}")

    if summary_rows:
        out_path = FAILURES_DIR / "logo_failure_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
