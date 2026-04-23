from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.utils.logo_artifacts import PROJECT_ROOT, REPORTS_DIR


FIGURES_DIR = REPORTS_DIR / "figures"
MPLCONFIGDIR = PROJECT_ROOT / ".mplconfig"


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    required = {"test_generator", "test_recall", "test_acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["test_recall"] = pd.to_numeric(df["test_recall"], errors="coerce")
    df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")
    return df


def _configure_matplotlib():
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to generate the figure set. "
            "Install it and re-run this script."
        ) from exc
    return plt


def _apply_style(plt) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def _save_recall_bar(df: pd.DataFrame, plt) -> Path:
    order = df.sort_values("test_recall", ascending=False)
    fig_path = FIGURES_DIR / "logo_recall_by_generator.png"

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(order["test_generator"], order["test_recall"], color="#1f6f8b")
    plt.axhline(order["test_recall"].mean(), color="#d1495b", linestyle="--", linewidth=1.5)
    plt.title("LOGO Recall by Held-out Generator")
    plt.xlabel("Held-out generator")
    plt.ylabel("Test recall")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=25, ha="right")
    for bar, value in zip(bars, order["test_recall"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()
    return fig_path


def _save_recall_accuracy_scatter(df: pd.DataFrame, plt) -> Path:
    fig_path = FIGURES_DIR / "logo_accuracy_vs_recall.png"

    plt.figure(figsize=(6.8, 5.4))
    plt.scatter(df["test_acc"], df["test_recall"], s=90, color="#2a9d8f")
    for _, row in df.iterrows():
        plt.text(
            row["test_acc"] + 0.001,
            row["test_recall"] + 0.001,
            str(row["test_generator"]),
            fontsize=8,
        )
    plt.axvline(df["test_acc"].mean(), color="#6c757d", linestyle="--", linewidth=1.2)
    plt.axhline(df["test_recall"].mean(), color="#d1495b", linestyle="--", linewidth=1.2)
    plt.title("LOGO Accuracy vs Recall")
    plt.xlabel("Test accuracy")
    plt.ylabel("Test recall")
    plt.xlim(0.84, 0.94)
    plt.ylim(0.78, 0.96)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()
    return fig_path


def _save_before_after_comparison(current_df: pd.DataFrame, baseline_df: pd.DataFrame, plt) -> Path | None:
    current = current_df[["test_generator", "test_recall", "test_acc"]].copy()
    baseline = baseline_df[["test_generator", "test_recall", "test_acc"]].copy()
    merged = current.merge(
        baseline,
        on="test_generator",
        suffixes=("_current", "_baseline"),
        how="inner",
    )
    if merged.empty:
        return None

    merged["recall_delta"] = merged["test_recall_current"] - merged["test_recall_baseline"]
    merged = merged.sort_values("recall_delta", ascending=False)

    fig_path = FIGURES_DIR / "logo_recall_before_vs_after.png"
    x = range(len(merged))
    width = 0.38

    plt.figure(figsize=(8.4, 4.8))
    plt.bar(
        [i - width / 2 for i in x],
        merged["test_recall_baseline"],
        width=width,
        label="5-generator baseline",
        color="#b8c0c8",
    )
    plt.bar(
        [i + width / 2 for i in x],
        merged["test_recall_current"],
        width=width,
        label="10-generator baseline",
        color="#0f766e",
    )
    plt.title("Recall Improvement on Overlapping Held-out Generators")
    plt.xlabel("Held-out generator")
    plt.ylabel("Test recall")
    plt.xticks(list(x), merged["test_generator"], rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()
    return fig_path


def _save_failure_counts(plt) -> Path | None:
    failure_path = REPORTS_DIR / "failures" / "logo_failure_summary.csv"
    if not failure_path.exists():
        return None

    df = pd.read_csv(failure_path)
    required = {"test_generator", "false_negatives", "false_positives"}
    if not required.issubset(df.columns):
        return None

    df = df.sort_values("false_negatives", ascending=False)
    fig_path = FIGURES_DIR / "logo_failure_counts.png"
    x = range(len(df))
    width = 0.36

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(
        [i - width / 2 for i in x],
        df["false_negatives"],
        width=width,
        label="False negatives",
        color="#bc4749",
    )
    plt.bar(
        [i + width / 2 for i in x],
        df["false_positives"],
        width=width,
        label="False positives",
        color="#457b9d",
    )
    plt.title("Failure Counts for Weakest LOGO Runs")
    plt.xlabel("Held-out generator")
    plt.ylabel("Count")
    plt.xticks(list(x), df["test_generator"], rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()
    return fig_path


def main() -> None:
    summary_path = REPORTS_DIR / "logo_summary.csv"
    baseline_5gen_path = REPORTS_DIR / "logo_summary_baseline_5gen.csv"

    current_df = _load_summary(summary_path)
    current_sorted = current_df.sort_values("test_recall", ascending=False).copy()
    current_sorted["macro_avg_test_recall"] = round(current_sorted["test_recall"].mean(), 6)
    current_sorted["macro_avg_test_acc"] = round(current_sorted["test_acc"].mean(), 6)

    enriched_path = REPORTS_DIR / "logo_summary_enriched.csv"
    current_sorted.to_csv(enriched_path, index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt = _configure_matplotlib()
    _apply_style(plt)

    saved_paths: list[Path] = []
    saved_paths.append(_save_recall_bar(current_sorted, plt))
    saved_paths.append(_save_recall_accuracy_scatter(current_sorted, plt))

    if baseline_5gen_path.exists():
        baseline_df = _load_summary(baseline_5gen_path)
        before_after_path = _save_before_after_comparison(current_sorted, baseline_df, plt)
        if before_after_path is not None:
            saved_paths.append(before_after_path)

    failure_plot_path = _save_failure_counts(plt)
    if failure_plot_path is not None:
        saved_paths.append(failure_plot_path)

    print(f"Macro-average test recall: {current_sorted['test_recall'].mean():.6f}")
    print(f"Macro-average test accuracy: {current_sorted['test_acc'].mean():.6f}")
    print(f"Saved: {enriched_path}")
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
