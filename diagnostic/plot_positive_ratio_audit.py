"""Build positive-ratio CSV/PNG audits from diagnostic runs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostic.constants import PER_GALLERY_RESULTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit positive ratios in per-gallery diagnostic results.")
    parser.add_argument("input_dirs", nargs="+", help="Diagnostic run output directories")
    parser.add_argument("--output_dir", required=True, help="Directory for audit outputs")
    parser.add_argument("--png_name", default="positive_ratio_audit.png")
    parser.add_argument("--csv_name", default="positive_ratio_audit.csv")
    return parser


def _read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["run_name", "dataset", "case_id", "gallery_type", "count", "mean_positive_ratio"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grouped = defaultdict(list)
    for run_dir in args.input_dirs:
        run_name = os.path.basename(os.path.abspath(run_dir))
        for row in _read_rows(os.path.join(run_dir, PER_GALLERY_RESULTS)):
            key = (run_name, row.get("dataset", ""), row.get("case_id", ""), row.get("gallery_type", ""))
            try:
                grouped[key].append(float(row.get("positive_ratio", 0.0) or 0.0))
            except ValueError:
                grouped[key].append(0.0)

    rows = []
    for (run_name, dataset, case_id, gallery_type), values in sorted(grouped.items()):
        mean_value = sum(values) / len(values) if values else 0.0
        rows.append(
            {
                "run_name": run_name,
                "dataset": dataset,
                "case_id": case_id,
                "gallery_type": gallery_type,
                "count": len(values),
                "mean_positive_ratio": mean_value,
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.csv_name)
    _write_rows(csv_path, rows)

    png_path = os.path.join(args.output_dir, args.png_name)
    try:
        import matplotlib.pyplot as plt

        labels = [f"{row['run_name']}:{row['gallery_type']}" for row in rows]
        values = [float(row["mean_positive_ratio"]) for row in rows]
        width = max(8, min(24, len(labels) * 0.35))
        fig, ax = plt.subplots(figsize=(width, 5))
        ax.bar(range(len(values)), values)
        ax.set_ylabel("Mean positive ratio")
        ax.set_ylim(0, max(values + [0.01]) * 1.2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
    except Exception:
        with open(png_path + ".skipped.txt", "w", encoding="utf-8") as handle:
            handle.write("matplotlib was unavailable or plotting failed; CSV audit was still written.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
