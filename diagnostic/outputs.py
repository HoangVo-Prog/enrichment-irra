"""Stable CSV, JSON, and JSONL output helpers."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Iterable

from diagnostic.config import json_safe
from diagnostic.constants import CSV_SCHEMAS, JSONL_SCHEMAS, OUTPUT_FILENAMES


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def output_path(output_dir: str, filename: str) -> str:
    return os.path.join(output_dir, filename)


def write_csv_rows(output_dir: str, filename: str, rows: Iterable[dict[str, Any]]) -> int:
    ensure_output_dir(output_dir)
    path = output_path(output_dir, filename)
    rows = list(rows)
    fieldnames = CSV_SCHEMAS.get(filename)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fieldnames})
    return len(rows)


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)


def write_jsonl_rows(output_dir: str, filename: str, rows: Iterable[dict[str, Any]]) -> int:
    ensure_output_dir(output_dir)
    path = output_path(output_dir, filename)
    rows = list(rows)
    schema = JSONL_SCHEMAS.get(filename)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            if schema is not None:
                row = {key: row.get(key) for key in schema if key in row}
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
    return len(rows)


def initialize_empty_outputs(output_dir: str) -> None:
    ensure_output_dir(output_dir)
    for filename in CSV_SCHEMAS:
        write_csv_rows(output_dir, filename, [])
    for filename in JSONL_SCHEMAS:
        write_jsonl_rows(output_dir, filename, [])


def write_all_outputs(output_dir: str, tables: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, filename in OUTPUT_FILENAMES.items():
        if filename.endswith(".csv"):
            counts[filename] = write_csv_rows(output_dir, filename, tables.get(key, []))
        elif filename.endswith(".jsonl"):
            counts[filename] = write_jsonl_rows(output_dir, filename, tables.get(key, []))
    return counts
