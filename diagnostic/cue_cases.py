"""Manual and automatic cue-case construction from query text."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable

from diagnostic.config import stable_int_seed
from diagnostic.cue_ontology import Cue
from diagnostic.data_loading import QueryRecord


@dataclass
class CueCase:
    case_id: str
    cue_a: str
    cue_b: str
    source: str
    query_ids: list[int]
    query_regex: str = ""


def _compile_optional(pattern: str) -> re.Pattern[str] | None:
    if not pattern:
        return None
    return re.compile(pattern, flags=re.IGNORECASE)


def _load_json_or_jsonl(path: str) -> list[dict]:
    if path.lower().endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("cases", [])
    if not isinstance(data, list):
        raise ValueError("Cases file must be a list, {'cases': [...]}, or JSONL")
    return data


def _matching_query_ids(
    queries: Iterable[QueryRecord],
    cue_a: Cue,
    cue_b: Cue,
    query_regex: str = "",
    explicit_query_ids: Iterable[int] | None = None,
) -> list[int]:
    valid_ids = {q.query_id for q in queries}
    if explicit_query_ids is not None:
        ids = [int(qid) for qid in explicit_query_ids if int(qid) in valid_ids]
        if query_regex:
            regex = _compile_optional(query_regex)
            by_id = {q.query_id: q for q in queries}
            ids = [qid for qid in ids if regex is None or regex.search(by_id[qid].text)]
        return sorted(set(ids))

    regex = _compile_optional(query_regex)
    cue_a_regex = cue_a.regex()
    cue_b_regex = cue_b.regex()
    ids = []
    for query in queries:
        text = query.text
        if regex is not None and not regex.search(text):
            continue
        if cue_a_regex.search(text) or cue_b_regex.search(text):
            ids.append(query.query_id)
    return ids


def load_manual_cases(path: str, queries: list[QueryRecord], cues: list[Cue]) -> list[CueCase]:
    cue_by_name = {cue.name: cue for cue in cues}
    rows = _load_json_or_jsonl(path)
    cases: list[CueCase] = []
    for idx, row in enumerate(rows):
        cue_a_name = str(row["cue_a"])
        cue_b_name = str(row["cue_b"])
        if cue_a_name not in cue_by_name or cue_b_name not in cue_by_name:
            raise ValueError(f"Unknown cue in manual case {idx}: {cue_a_name!r}, {cue_b_name!r}")
        query_regex = str(row.get("query_regex", ""))
        query_ids = _matching_query_ids(
            queries,
            cue_by_name[cue_a_name],
            cue_by_name[cue_b_name],
            query_regex=query_regex,
            explicit_query_ids=row.get("query_ids"),
        )
        case_id = str(row.get("case_id") or f"manual_{idx:04d}")
        cases.append(CueCase(case_id, cue_a_name, cue_b_name, "manual", query_ids, query_regex))
    return cases


def generate_auto_cases(
    queries: list[QueryRecord],
    cues: list[Cue],
    min_queries_per_case: int,
    max_cases: int | None,
) -> list[CueCase]:
    cases: list[CueCase] = []
    by_category: dict[str, list[Cue]] = {}
    for cue in cues:
        by_category.setdefault(cue.category, []).append(cue)

    for category, group in sorted(by_category.items()):
        for i, cue_a in enumerate(group):
            for cue_b in group[i + 1 :]:
                query_ids = _matching_query_ids(queries, cue_a, cue_b)
                if len(query_ids) < min_queries_per_case:
                    continue
                seed = stable_int_seed(category, cue_a.name, cue_b.name)
                case_id = f"auto_{category}_{seed:08x}"
                cases.append(CueCase(case_id, cue_a.name, cue_b.name, "auto", query_ids, ""))

    cases.sort(key=lambda case: (-len(case.query_ids), case.case_id))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def build_cases(
    queries: list[QueryRecord],
    cues: list[Cue],
    cases_file: str | None,
    auto_cases: bool,
    min_queries_per_auto_case: int,
    max_auto_cases: int | None,
) -> list[CueCase]:
    cases: list[CueCase] = []
    if cases_file:
        if not os.path.exists(cases_file):
            raise FileNotFoundError(f"Cases file not found: {cases_file}")
        cases.extend(load_manual_cases(cases_file, queries, cues))

    should_auto = auto_cases or not cases_file
    if should_auto:
        cases.extend(
            generate_auto_cases(
                queries,
                cues,
                min_queries_per_case=min_queries_per_auto_case,
                max_cases=max_auto_cases,
            )
        )
    return cases


def select_queries_for_cases(
    cases: list[CueCase],
    queries: list[QueryRecord],
    max_queries_per_case: int | None,
    seed: int,
) -> tuple[list[dict], dict[str, list[QueryRecord]]]:
    import random

    by_id = {query.query_id: query for query in queries}
    candidate_rows = []
    selected: dict[str, list[QueryRecord]] = {}
    for case in cases:
        ids = [qid for qid in case.query_ids if qid in by_id]
        if max_queries_per_case is not None and len(ids) > max_queries_per_case:
            rng = random.Random(stable_int_seed(seed, case.case_id, "query_selection"))
            ids = sorted(rng.sample(ids, max_queries_per_case))
        selected[case.case_id] = [by_id[qid] for qid in ids]
        candidate_rows.append(
            {
                "case_id": case.case_id,
                "cue_a": case.cue_a,
                "cue_b": case.cue_b,
                "source": case.source,
                "num_queries": len(case.query_ids),
                "query_regex": case.query_regex,
            }
        )
    return candidate_rows, selected
