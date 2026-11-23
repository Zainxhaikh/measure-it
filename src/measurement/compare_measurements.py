
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    PROJECT_ROOT
    / "results"
    / "measurement"
    / "manual_comparison.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "results"
    / "measurement"
    / "measurement_comparison_report.txt"
)


def calculate_metrics(reference, predicted):
    errors = predicted - reference
    absolute_errors = np.abs(errors)
    percentage_errors = (
        absolute_errors / reference
    ) * 100

    return {
        "MAE": np.mean(absolute_errors),
        "Mean Error": np.mean(errors),
        "Median Absolute Error": np.median(absolute_errors),
        "Worst-case Error": np.max(absolute_errors),
        "Mean Percentage Error": np.mean(percentage_errors),
    }


def main():
    print("=" * 60)
    print("CARD MEASUREMENT COMPARISON")
    print("=" * 60)

    if not CSV_FILE.exists():
        print(f"CSV file not found: {CSV_FILE}")
        return

    data = pd.read_csv(CSV_FILE)

    required_columns = [
        "image_name",
        "reference_width_mm",
        "ai_width_mm",
        "reference_height_mm",
        "ai_height_mm",
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        print("Missing CSV columns:", missing)
        return

    if data[required_columns].isnull().any().any():
        print("Error: CSV contains empty measurement values.")
        return

    if len(data) < 10:
        print("Error: At least 10 images are required.")
        return

    width = calculate_metrics(
        data["reference_width_mm"].to_numpy(),
        data["ai_width_mm"].to_numpy(),
    )

    height = calculate_metrics(
        data["reference_height_mm"].to_numpy(),
        data["ai_height_mm"].to_numpy(),
    )

    lines = []

    lines.append("CARD MEASUREMENT COMPARISON REPORT")
    lines.append("=" * 50)
    lines.append(f"Images evaluated: {len(data)}")
    lines.append("Reference dimensions: 57 x 86 mm")
    lines.append("")

    for dimension, metrics in [
        ("WIDTH", width),
        ("HEIGHT", height),
    ]:
        lines.append(f"{dimension} RESULTS")
        lines.append("-" * 40)

        for metric, value in metrics.items():
            lines.append(f"{metric}: {value:.4f}")

        lines.append("")

    lines.append("INDIVIDUAL IMAGE COMPARISON")
    lines.append("-" * 50)

    for _, row in data.iterrows():
        width_error = abs(
            row["ai_width_mm"] - row["reference_width_mm"]
        )

        height_error = abs(
            row["ai_height_mm"] - row["reference_height_mm"]
        )

        lines.append(
            f"{row['image_name']}: "
            f"Width error = {width_error:.2f} mm, "
            f"Height error = {height_error:.2f} mm"
        )

    report = "\n".join(lines)

    print(report)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")

    print("\nReport saved to:")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()
