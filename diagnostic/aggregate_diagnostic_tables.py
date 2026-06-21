"""Aggregate diagnostic output folders into publication-oriented tables."""

from __future__ import annotations

import argparse
import csv
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostic.constants import (
    SUMMARY_BY_CASE,
    SUMMARY_OVERALL,
    SUMMARY_WITH_CI,
    SUMMARY_WITH_CI_CASE_QUERY,
    SUMMARY_WITH_CI_UNIQUE_QUERY,
    VALIDITY_COUNTS,
)


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate cue-swap diagnostic output folders.")
    parser.add_argument("input_dirs", nargs="+", help="Diagnostic run output directories")
    parser.add_argument("--output_dir", required=True, help="Directory for aggregate CSV tables")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generalized = []
    hardness = []
    validity = []
    bootstrap_ci = []
    for run_dir in args.input_dirs:
        run_name = os.path.basename(os.path.abspath(run_dir))
        for row in _read_csv(os.path.join(run_dir, SUMMARY_OVERALL)):
            row = {"run_dir": run_dir, "run_name": run_name, **row}
            generalized.append(row)
            hardness.append(
                {
                    key: row.get(key, "")
                    for key in (
                        "run_dir",
                        "run_name",
                        "dataset",
                        "retriever_name",
                        "cue_scorer",
                        "num_pairs",
                        "hm_r1_flip",
                        "hm_rank_shift",
                        "delta_r1_flip",
                        "delta_rank_shift",
                    )
                }
            )
        for row in _read_csv(os.path.join(run_dir, SUMMARY_BY_CASE)):
            generalized.append({"run_dir": run_dir, "run_name": run_name, **row})
        for row in _read_csv(os.path.join(run_dir, VALIDITY_COUNTS)):
            validity.append({"run_dir": run_dir, "run_name": run_name, **row})
        seen_ci_units = set()
        for filename in (SUMMARY_WITH_CI_UNIQUE_QUERY, SUMMARY_WITH_CI_CASE_QUERY, SUMMARY_WITH_CI):
            for row in _read_csv(os.path.join(run_dir, filename)):
                unit = row.get("bootstrap_unit", "")
                if filename == SUMMARY_WITH_CI and unit in seen_ci_units:
                    continue
                if filename != SUMMARY_WITH_CI and unit:
                    seen_ci_units.add(unit)
                bootstrap_ci.append({"run_dir": run_dir, "run_name": run_name, "source_file": filename, **row})

    _write_csv(os.path.join(args.output_dir, "generalized_cue_swap_table.csv"), generalized)
    _write_csv(os.path.join(args.output_dir, "hardness_control_table.csv"), hardness)
    _write_csv(os.path.join(args.output_dir, "validity_counts_table.csv"), validity)
    _write_csv(os.path.join(args.output_dir, "bootstrap_ci_table.csv"), bootstrap_ci)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
