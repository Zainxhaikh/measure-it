
from pathlib import Path
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    PROJECT_ROOT / "results" / "measurement" /
    "manual_comparison.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "results" / "measurement" /
    "measurement_summary.json"
)


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError("Measurement comparison CSV not found.")

    results = []

    with CSV_FILE.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if not row["ai_width_mm"] or not row["ai_height_mm"]:
                continue

            results.append({
                "image_name": row["image_name"],
                "ai_width_mm": float(row["ai_width_mm"]),
                "ai_height_mm": float(row["ai_height_mm"]),
                "manual_width_available": bool(row["manual_width_mm"]),
                "manual_height_available": bool(row["manual_height_mm"]),
            })

    summary = {
        "total_successful_ai_measurements": len(results),
        "measurements": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("MEASUREMENT SUMMARY")
    print("===================")
    print(f"Successful AI measurements: {len(results)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
