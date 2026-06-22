"""Manual and automatic cue-case construction from query text."""

from __future__ import annotations

import json
import logging
import os
import re
import time
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


def _normalize_query_text(text: str) -> str:
    return text.casefold()


def _cue_pattern_text(cue: Cue) -> str:
    joined = "|".join(cue.patterns)
    return rf"\b(?:{joined})\b"


def _fallback_cue(name: str) -> Cue:
    compact_space = re.escape(name).replace(r"\ ", r"\s+")
    hyphen_as_space = re.escape(name.replace("-", " ")).replace(r"\ ", r"[\s-]+")
    patterns = tuple(dict.fromkeys((compact_space, hyphen_as_space)))
    return Cue(name=name, category="manual_case", patterns=patterns)


def _strip_leading_anywhere(pattern: str) -> str:
    if pattern.startswith(".*?"):
        return pattern[3:]
    if pattern.startswith(".*"):
        return pattern[2:]
    return pattern


def _positive_lookahead_fragments(pattern: str) -> tuple[str, ...] | None:
    text = pattern.strip()
    if not text:
        return None

    fragments: list[str] = []
    position = 0
    while position < len(text):
        if not text.startswith("(?=", position):
            return None
        depth = 1
        index = position + 3
        while index < len(text):
            char = text[index]
            if char == "\\":
                index += 2
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    fragments.append(_strip_leading_anywhere(text[position + 3 : index]))
                    position = index + 1
                    break
            index += 1
        else:
            return None

    return tuple(fragment for fragment in fragments if fragment) or None


class _QueryTextIndex:
    def __init__(self, queries: Iterable[QueryRecord]):
        query_list = list(queries)
        self.by_id = {query.query_id: query for query in query_list}
        self.valid_ids = frozenset(self.by_id)
        self._normalized_text = {
            query.query_id: _normalize_query_text(query.text)
            for query in query_list
        }
        self._regex_cache: dict[str, frozenset[int]] = {}
        self._lookahead_fragment_cache: dict[str, tuple[str, ...] | None] = {}

    @property
    def cached_regex_fragments(self) -> int:
        return len(self._regex_cache)

    def _compile(self, pattern: str) -> re.Pattern[str]:
        return re.compile(pattern, flags=re.IGNORECASE)

    def match_regex(self, pattern: str) -> frozenset[int]:
        if not pattern:
            return self.valid_ids
        if pattern not in self._regex_cache:
            regex = self._compile(pattern)
            self._regex_cache[pattern] = frozenset(
                query_id
                for query_id, text in self._normalized_text.items()
                if regex.search(text)
            )
        return self._regex_cache[pattern]

    def match_query_regex(self, pattern: str) -> frozenset[int]:
        if not pattern:
            return self.valid_ids
        if pattern not in self._lookahead_fragment_cache:
            self._lookahead_fragment_cache[pattern] = _positive_lookahead_fragments(pattern)
        fragments = self._lookahead_fragment_cache[pattern]
        if not fragments:
            return self.match_regex(pattern)

        matches: frozenset[int] | None = None
        for fragment in fragments:
            fragment_matches = self.match_regex(fragment)
            matches = fragment_matches if matches is None else matches & fragment_matches
            if not matches:
                break
        return matches if matches is not None else frozenset()

    def match_any(self, patterns: Iterable[str]) -> frozenset[int]:
        matches: set[int] = set()
        for pattern in patterns:
            matches.update(self.match_regex(pattern))
        return frozenset(matches)


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
    text_index: _QueryTextIndex | None = None,
) -> list[int]:
    index = text_index or _QueryTextIndex(queries)
    valid_ids = index.valid_ids
    if explicit_query_ids is not None:
        ids = [int(qid) for qid in explicit_query_ids if int(qid) in valid_ids]
        if query_regex:
            regex_ids = index.match_query_regex(query_regex)
            ids = [qid for qid in ids if qid in regex_ids]
        return sorted(set(ids))

    regex_ids = index.match_query_regex(query_regex) if query_regex else valid_ids
    if query_regex:
        return sorted(regex_ids)
    cue_ids = index.match_any((_cue_pattern_text(cue_a), _cue_pattern_text(cue_b)))
    return sorted(regex_ids & cue_ids)


def load_manual_cases(
    path: str,
    queries: list[QueryRecord],
    cues: list[Cue],
    text_index: _QueryTextIndex | None = None,
    logger: logging.Logger | None = None,
) -> list[CueCase]:
    cue_by_name = {cue.name: cue for cue in cues}
    rows = _load_json_or_jsonl(path)
    unknown_cues = sorted(
        {
            str(row.get(field, ""))
            for row in rows
            for field in ("cue_a", "cue_b")
            if str(row.get(field, "")) and str(row.get(field, "")) not in cue_by_name
        }
    )
    if unknown_cues:
        for cue_name in unknown_cues:
            cue_by_name[cue_name] = _fallback_cue(cue_name)
        if logger is not None:
            shown = ", ".join(repr(cue) for cue in unknown_cues[:12])
            hidden = len(unknown_cues) - min(len(unknown_cues), 12)
            suffix = f", ... {hidden} more" if hidden else ""
            logger.warning(
                "Manual cases reference %d cues outside the loaded ontology; "
                "using case query_regex/query_ids for selection where available and retaining cue names for CLIP scoring. "
                "Examples: %s%s",
                len(unknown_cues),
                shown,
                suffix,
            )
    index = text_index or _QueryTextIndex(queries)
    cases: list[CueCase] = []
    start = time.time()
    for idx, row in enumerate(rows):
        cue_a_name = str(row["cue_a"])
        cue_b_name = str(row["cue_b"])
        query_regex = str(row.get("query_regex", ""))
        query_ids = _matching_query_ids(
            queries,
            cue_by_name[cue_a_name],
            cue_by_name[cue_b_name],
            query_regex=query_regex,
            explicit_query_ids=row.get("query_ids"),
            text_index=index,
        )
        case_id = str(row.get("case_id") or f"manual_{idx:04d}")
        cases.append(CueCase(case_id, cue_a_name, cue_b_name, "manual", query_ids, query_regex))
        if logger is not None and ((idx + 1) % 250 == 0 or idx + 1 == len(rows)):
            logger.info(
                "Manual case matching progress cases=%d/%d cached_regex_fragments=%d elapsed=%.1fs",
                idx + 1,
                len(rows),
                index.cached_regex_fragments,
                time.time() - start,
            )
    return cases


def generate_auto_cases(
    queries: list[QueryRecord],
    cues: list[Cue],
    min_queries_per_case: int,
    max_cases: int | None,
    text_index: _QueryTextIndex | None = None,
) -> list[CueCase]:
    index = text_index or _QueryTextIndex(queries)
    cases: list[CueCase] = []
    by_category: dict[str, list[Cue]] = {}
    for cue in cues:
        by_category.setdefault(cue.category, []).append(cue)

    for category, group in sorted(by_category.items()):
        for i, cue_a in enumerate(group):
            for cue_b in group[i + 1 :]:
                query_ids = _matching_query_ids(queries, cue_a, cue_b, text_index=index)
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
    logger: logging.Logger | None = None,
) -> list[CueCase]:
    cases: list[CueCase] = []
    text_index = _QueryTextIndex(queries)
    if cases_file:
        if not os.path.exists(cases_file):
            raise FileNotFoundError(f"Cases file not found: {cases_file}")
        cases.extend(load_manual_cases(cases_file, queries, cues, text_index=text_index, logger=logger))

    should_auto = auto_cases or not cases_file
    if should_auto:
        cases.extend(
            generate_auto_cases(
                queries,
                cues,
                min_queries_per_case=min_queries_per_auto_case,
                max_cases=max_auto_cases,
                text_index=text_index,
            )
        )
    return cases


def select_queries_for_cases(
    cases: list[CueCase],
    queries: list[QueryRecord],
    max_queries_per_case: int | None,
    seed: int,
    logger: logging.Logger | None = None,
) -> tuple[list[dict], dict[str, list[QueryRecord]]]:
    import random

    text_index = _QueryTextIndex(queries)
    by_id = text_index.by_id
    candidate_rows = []
    selected: dict[str, list[QueryRecord]] = {}
    skipped_rows = 0
    start = time.time()
    selected_query_count = 0
    for idx, case in enumerate(cases, start=1):
        regex_ids = text_index.match_query_regex(case.query_regex) if case.query_regex else text_index.valid_ids
        ids = [qid for qid in case.query_ids if qid in by_id and qid in regex_ids]
        skipped_rows += max(0, len(case.query_ids) - len(ids))
        if max_queries_per_case is not None and len(ids) > max_queries_per_case:
            rng = random.Random(stable_int_seed(seed, case.case_id, "query_selection"))
            ids = sorted(rng.sample(ids, max_queries_per_case))
        selected[case.case_id] = [by_id[qid] for qid in ids]
        selected_query_count += len(ids)
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
        if logger is not None and (idx % 250 == 0 or idx == len(cases)):
            logger.info(
                "Query selection progress cases=%d/%d selected_queries=%d skipped_rows=%d "
                "cached_regex_fragments=%d elapsed=%.1fs",
                idx,
                len(cases),
                selected_query_count,
                skipped_rows,
                text_index.cached_regex_fragments,
                time.time() - start,
            )
    return candidate_rows, selected
