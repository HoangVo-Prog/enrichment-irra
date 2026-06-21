"""Main CLI for the IRRA cue-swap diagnostic."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import Counter, defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from diagnostic.audit import audit_outputs
from diagnostic.bootstrap import (
    bootstrap_count_stats,
    bootstrap_unit_label,
    cluster_bootstrap_by_unit,
    cluster_columns_for_unit,
)
from diagnostic.clip_cue_scorer import OffTheShelfCLIPCueScorer, threshold_rows
from diagnostic.config import (
    build_parser,
    config_payload,
    load_irra_config,
    resolve_device,
    set_deterministic,
    validate_args,
)
from diagnostic.constants import (
    BOOTSTRAP_UNIT_CASE_QUERY,
    BOOTSTRAP_UNIT_UNIQUE_QUERY,
    CONFIG_USED,
    GALLERY_TYPE_CUE_A,
    GALLERY_TYPE_CUE_B,
    GALLERY_TYPE_HM_A,
    GALLERY_TYPE_HM_B,
    OUTPUT_FILENAMES,
    SUMMARY_WITH_CI_CASE_QUERY,
    SUMMARY_WITH_CI_UNIQUE_QUERY,
)
from diagnostic.controls import construct_hardness_controls
from diagnostic.cue_cases import build_cases, select_queries_for_cases
from diagnostic.cue_ontology import load_cues
from diagnostic.data_loading import empty_split, load_split
from diagnostic.embeddings import extract_retrieval_embeddings
from diagnostic.gallery_construction import construct_gallery_pair, constructibility_rows
from diagnostic.metrics import cue_density, cue_shift, paired_retrieval_metrics, positive_ratio, retrieval_metrics
from diagnostic.outputs import ensure_output_dir, output_path, write_all_outputs, write_csv_rows, write_json
from diagnostic.retriever_loading import load_retriever
from diagnostic.scoring import precompute_score_cache, resolve_score_mode, score_gallery, whole_test_metrics


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    return logging.getLogger("diagnostic.run")


def _empty_tables() -> dict[str, list[dict]]:
    return {key: [] for key in OUTPUT_FILENAMES if key != "config_used"}


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _format_table(rows: list[dict], columns: list[str]) -> str:
    def format_value(value):
        if value is None:
            return ""
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            value = float(value)
            if not np.isfinite(value):
                return str(value)
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    rendered_rows = [[format_value(row.get(column, "")) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(row[index]) for row in rendered_rows)) if rendered_rows else len(column)
        for index, column in enumerate(columns)
    ]
    lines = [" ".join(column.rjust(widths[index]) for index, column in enumerate(columns))]
    lines.extend(" ".join(value.rjust(widths[index]) for index, value in enumerate(row)) for row in rendered_rows)
    return "\n".join(lines)


def _log_final_report(logger: logging.Logger, output_dir: str, summary_rows: list[dict], ci_rows: list[dict]) -> None:
    summary_columns = [
        "dataset",
        "retriever_name",
        "cue_scorer",
        "ref_R1",
        "num_cases",
        "num_queries",
        "num_pairs",
        "valid_pair_rate",
        "mean_cue_shift",
        "r1_flip",
        "rank_shift",
        "hm_r1_flip",
        "hm_rank_shift",
        "delta_r1_flip",
        "delta_rank_shift",
    ]
    ci_columns = [
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "bootstrap_iters",
        "bootstrap_unit",
        "cluster_count",
        "unique_query_count",
        "case_query_count",
        "trial_count",
    ]
    print(
        "Summary overall:\n"
        f"{_format_table(summary_rows, summary_columns)}\n\n"
        "Bootstrap CIs:\n"
        f"{_format_table(ci_rows, ci_columns)}",
        flush=True,
    )
    logger.info("Wrote diagnostic outputs to %s", output_dir)


def _selected_query_rows(args, cases, selected_by_case, split_data):
    rows = []
    gallery_pids = split_data.gallery_pids
    for case in cases:
        for query in selected_by_case.get(case.case_id, []):
            rows.append(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "retriever_name": args.retriever_name,
                    "case_id": case.case_id,
                    "cue_a": case.cue_a,
                    "cue_b": case.cue_b,
                    "query_id": query.query_id,
                    "pid": query.pid,
                    "text": query.text,
                    "num_positives": int((gallery_pids == int(query.pid)).sum()),
                    "num_distractors": int((gallery_pids != int(query.pid)).sum()),
                }
            )
    return rows


def _gallery_metrics_row(
    args,
    case,
    query,
    trial_id,
    gallery_type,
    gallery_ids,
    score_vector,
    split_data,
    density_a,
    density_b,
    shift_value,
    ref_rank,
    score_mode,
):
    ids = list(gallery_ids)
    scores = score_gallery(score_vector, ids)
    pids = split_data.gallery_pids[np.asarray(ids, dtype=np.int64)]
    metrics = retrieval_metrics(scores, pids, query.pid, image_ids=ids)
    return {
        "dataset": args.dataset,
        "split": args.split,
        "retriever_name": args.retriever_name,
        "cue_scorer": args.cue_scorer,
        "case_id": case.case_id,
        "cue_a": case.cue_a,
        "cue_b": case.cue_b,
        "query_id": query.query_id,
        "pid": query.pid,
        "trial_id": trial_id,
        "gallery_type": gallery_type,
        "num_gallery": len(ids),
        "num_positives": int((pids == int(query.pid)).sum()),
        "num_distractors": int((pids != int(query.pid)).sum()),
        "R1": metrics["R1"],
        "R5": metrics["R5"],
        "R10": metrics["R10"],
        "AP": metrics["AP"],
        "best_positive_rank": metrics["best_positive_rank"],
        "positive_ratio": positive_ratio(ids, query.pid, split_data.gallery_pids),
        "cue_density_a": density_a,
        "cue_density_b": density_b,
        "cue_shift": shift_value,
        "ref_best_positive_rank": ref_rank,
        "score_mode": score_mode,
    }


def _paired_rows(args, case, query, trial_id, pair, a_row, b_row, hm_a_row, hm_b_row, hm, shift_value):
    cue_pair = paired_retrieval_metrics(
        {"R1": a_row["R1"], "AP": a_row["AP"], "best_positive_rank": a_row["best_positive_rank"]},
        {"R1": b_row["R1"], "AP": b_row["AP"], "best_positive_rank": b_row["best_positive_rank"]},
    )
    hm_pair = paired_retrieval_metrics(
        {"R1": hm_a_row["R1"], "AP": hm_a_row["AP"], "best_positive_rank": hm_a_row["best_positive_rank"]},
        {"R1": hm_b_row["R1"], "AP": hm_b_row["AP"], "best_positive_rank": hm_b_row["best_positive_rank"]},
    )
    base = {
        "dataset": args.dataset,
        "split": args.split,
        "retriever_name": args.retriever_name,
        "cue_scorer": args.cue_scorer,
        "case_id": case.case_id,
        "cue_a": case.cue_a,
        "cue_b": case.cue_b,
        "query_id": query.query_id,
        "pid": query.pid,
        "trial_id": trial_id,
    }
    cue_row = {
        **base,
        "gallery_size": len(pair.gallery_a),
        "cue_shift": shift_value,
        "a_R1": a_row["R1"],
        "b_R1": b_row["R1"],
        "r1_flip": cue_pair["r1_flip"],
        "a_best_positive_rank": a_row["best_positive_rank"],
        "b_best_positive_rank": b_row["best_positive_rank"],
        "rank_shift": cue_pair["rank_shift"],
        "a_AP": a_row["AP"],
        "b_AP": b_row["AP"],
        "ap_delta": cue_pair["ap_delta"],
    }
    hm_row = {
        **base,
        "gallery_size": len(hm.hm_a),
        "hm_a_R1": hm_a_row["R1"],
        "hm_b_R1": hm_b_row["R1"],
        "hm_r1_flip": hm_pair["r1_flip"],
        "hm_a_best_positive_rank": hm_a_row["best_positive_rank"],
        "hm_b_best_positive_rank": hm_b_row["best_positive_rank"],
        "hm_rank_shift": hm_pair["rank_shift"],
        "hm_a_AP": hm_a_row["AP"],
        "hm_b_AP": hm_b_row["AP"],
        "hm_ap_delta": hm_pair["ap_delta"],
        **hm.diagnostics,
    }
    delta_row = {
        **base,
        "r1_flip": cue_pair["r1_flip"],
        "hm_r1_flip": hm_pair["r1_flip"],
        "delta_r1_flip": cue_pair["r1_flip"] - hm_pair["r1_flip"],
        "rank_shift": cue_pair["rank_shift"],
        "hm_rank_shift": hm_pair["rank_shift"],
        "delta_rank_shift": cue_pair["rank_shift"] - hm_pair["rank_shift"],
        "ap_delta": cue_pair["ap_delta"],
        "hm_ap_delta": hm_pair["ap_delta"],
        "delta_ap_delta": cue_pair["ap_delta"] - hm_pair["ap_delta"],
        "cue_shift": shift_value,
    }
    return cue_row, hm_row, delta_row


def _summary_rows(args, whole_row, cases, selected_by_case, delta_rows, attempted_by_case, attempted_total):
    ref_r1 = float(whole_row.get("R1", 0.0)) if whole_row else 0.0
    num_pairs = len(delta_rows)
    valid_rate = num_pairs / attempted_total if attempted_total else 0.0
    overall = [
        {
            "dataset": args.dataset,
            "retriever_name": args.retriever_name,
            "cue_scorer": args.cue_scorer,
            "ref_R1": ref_r1,
            "num_cases": len(cases),
            "num_queries": len({row["query_id"] for row in delta_rows}),
            "num_pairs": num_pairs,
            "valid_pair_rate": valid_rate,
            "mean_cue_shift": _mean(delta_rows, "cue_shift"),
            "r1_flip": _mean(delta_rows, "r1_flip"),
            "rank_shift": _mean(delta_rows, "rank_shift"),
            "hm_r1_flip": _mean(delta_rows, "hm_r1_flip"),
            "hm_rank_shift": _mean(delta_rows, "hm_rank_shift"),
            "delta_r1_flip": _mean(delta_rows, "delta_r1_flip"),
            "delta_rank_shift": _mean(delta_rows, "delta_rank_shift"),
        }
    ]

    by_case = []
    for case in cases:
        rows = [row for row in delta_rows if row.get("case_id") == case.case_id]
        attempted = attempted_by_case.get(case.case_id, 0)
        by_case.append(
            {
                "dataset": args.dataset,
                "retriever_name": args.retriever_name,
                "cue_scorer": args.cue_scorer,
                "case_id": case.case_id,
                "cue_a": case.cue_a,
                "cue_b": case.cue_b,
                "num_queries": len(selected_by_case.get(case.case_id, [])),
                "num_pairs": len(rows),
                "valid_pair_rate": len(rows) / attempted if attempted else 0.0,
                "mean_cue_shift": _mean(rows, "cue_shift"),
                "r1_flip": _mean(rows, "r1_flip"),
                "rank_shift": _mean(rows, "rank_shift"),
                "hm_r1_flip": _mean(rows, "hm_r1_flip"),
                "hm_rank_shift": _mean(rows, "hm_rank_shift"),
                "delta_r1_flip": _mean(rows, "delta_r1_flip"),
                "delta_rank_shift": _mean(rows, "delta_rank_shift"),
            }
        )
    return overall, by_case


def _gallery_json_rows(args, case, query, trial_id, galleries, split_data, save_paths):
    rows = []
    for gallery_type, image_ids in galleries.items():
        row = {
            "dataset": args.dataset,
            "split": args.split,
            "retriever_name": args.retriever_name,
            "case_id": case.case_id,
            "query_id": query.query_id,
            "trial_id": trial_id,
            "gallery_type": gallery_type,
            "image_ids": list(map(int, image_ids)),
            "image_paths": [],
        }
        if save_paths:
            row["image_paths"] = [split_data.gallery_paths[int(i)] for i in image_ids]
        rows.append(row)
    return rows


def _candidate_stats(values: list[float]) -> str:
    if not values:
        return "none"
    arr = np.asarray(values, dtype=np.float64)
    return f"min={arr.min():.4f}, mean={arr.mean():.4f}, max={arr.max():.4f}"


def _requested_bootstrap_units(bootstrap_unit: str) -> list[str]:
    if bootstrap_unit == "both":
        return [BOOTSTRAP_UNIT_UNIQUE_QUERY, BOOTSTRAP_UNIT_CASE_QUERY]
    return [bootstrap_unit]


def _primary_bootstrap_unit(bootstrap_unit: str) -> str:
    if bootstrap_unit == "both":
        return BOOTSTRAP_UNIT_UNIQUE_QUERY
    return bootstrap_unit


def _bootstrap_filename(unit: str) -> str:
    if unit == BOOTSTRAP_UNIT_UNIQUE_QUERY:
        return SUMMARY_WITH_CI_UNIQUE_QUERY
    if unit == BOOTSTRAP_UNIT_CASE_QUERY:
        return SUMMARY_WITH_CI_CASE_QUERY
    raise ValueError(f"Unsupported bootstrap unit: {unit}")


def _run_bootstraps(rows, args, logger):
    results: dict[str, list[dict]] = {}
    for unit in _requested_bootstrap_units(args.bootstrap_unit):
        cluster_columns = cluster_columns_for_unit(unit)
        stats = bootstrap_count_stats(rows, cluster_columns)
        logger.info(
            "Starting %s:\n"
            "unique_query_count=%d\n"
            "case_query_count=%d\n"
            "cluster_count=%d\n"
            "trial_count=%d\n"
            "bootstrap_iters=%d\n"
            "bootstrap_seed=%d",
            bootstrap_unit_label(unit),
            stats["unique_query_count"],
            stats["case_query_count"],
            stats["cluster_count"],
            stats["trial_count"],
            args.bootstrap_iters,
            args.bootstrap_seed,
        )
        boot_start = time.time()
        results[unit] = cluster_bootstrap_by_unit(
            rows,
            iters=args.bootstrap_iters,
            seed=args.bootstrap_seed,
            bootstrap_unit=unit,
        )
        logger.info("Completed %s: elapsed=%.1fs", bootstrap_unit_label(unit), time.time() - boot_start)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging()
    validate_args(args)
    set_deterministic(args.seed, include_torch=not args.dry_run)
    resolved_device = resolve_device(args.device, dry_run=args.dry_run)
    ensure_output_dir(args.output_dir)

    logger.info("Run parameters: %s", vars(args))
    logger.info("Resolved device: %s", resolved_device)

    irra_args = load_irra_config(args)
    write_json(output_path(args.output_dir, CONFIG_USED), config_payload(args, irra_args, resolved_device))

    tables = _empty_tables()
    split_data = empty_split()
    try:
        split_data = load_split(irra_args, args.split, metadata_only=args.dry_run)
        logger.info(
            "Loaded %s/%s metadata: queries=%d gallery=%d",
            args.dataset,
            args.split,
            len(split_data.query_records),
            len(split_data.gallery_records),
        )
    except Exception as exc:
        if not args.dry_run:
            raise
        logger.warning("Dry run could not load dataset metadata: %s", exc)

    cues = load_cues(args.cue_vocab_file)
    cases = build_cases(
        split_data.query_records,
        cues,
        cases_file=args.cases_file,
        auto_cases=args.auto_cases,
        min_queries_per_auto_case=args.min_queries_per_auto_case,
        max_auto_cases=args.max_auto_cases,
    )
    candidate_rows, selected_by_case = select_queries_for_cases(
        cases,
        split_data.query_records,
        args.max_queries_per_case,
        args.seed,
    )
    tables["cue_case_candidates"] = candidate_rows
    tables["cue_case_constructibility"] = constructibility_rows(
        cases,
        selected_by_case,
        split_data.gallery_pids,
        args.gallery_size,
        args.dense_ratio,
    )
    tables["selected_queries"] = _selected_query_rows(args, cases, selected_by_case, split_data)
    logger.info("Cue cases: candidates=%d retained=%d", len(candidate_rows), len(cases))
    logger.info("Selected query rows: %d", len(tables["selected_queries"]))

    if args.dry_run:
        bootstrap_results = _run_bootstraps(tables["paired_delta_results"], args, logger)
        primary_unit = _primary_bootstrap_unit(args.bootstrap_unit)
        tables["summary_with_ci"] = bootstrap_results[primary_unit]
        row_counts = write_all_outputs(args.output_dir, tables)
        for unit, rows in bootstrap_results.items():
            filename = _bootstrap_filename(unit)
            row_counts[filename] = write_csv_rows(args.output_dir, filename, rows)
        logger.info("Dry run complete. Output row counts: %s", row_counts)
        logger.info("Wrote diagnostic outputs to %s", args.output_dir)
        return 0

    score_mode = resolve_score_mode(args.score_mode)
    retriever = load_retriever(irra_args, split_data.num_classes, args.retriever_checkpoint, resolved_device)
    text_embeddings, image_embeddings = extract_retrieval_embeddings(retriever, split_data)
    whole = whole_test_metrics(text_embeddings.embeddings, image_embeddings.embeddings, split_data.query_pids, split_data.gallery_pids)
    whole_row = {
        "dataset": args.dataset,
        "split": args.split,
        "retriever_name": args.retriever_name,
        "score_mode": score_mode,
        **whole,
    }
    tables["whole_test_metrics"] = [whole_row]
    logger.info("Whole-test sanity metrics: %s", whole_row)

    unique_cues = sorted({case.cue_a for case in cases} | {case.cue_b for case in cases})
    if unique_cues:
        cue_scorer = OffTheShelfCLIPCueScorer(
            args.clip_model_name,
            resolved_device,
            image_size=irra_args.img_size,
            stride_size=irra_args.stride_size,
        )
        cue_result = cue_scorer.score(unique_cues, split_data.img_loader, text_length=irra_args.text_length, logger=logger)
        cue_scores = cue_result.scores
        cue_thresholds, tables["cue_thresholds"] = threshold_rows(
            cue_scores,
            cue_result.prompts,
            args.cue_threshold_quantile,
        )
        logger.info("Cue threshold stats written for %d cues", len(tables["cue_thresholds"]))
    else:
        cue_scores = {}
        cue_thresholds = {}
        logger.warning("No cue cases were available; skipping cue scoring")

    selected_items = []
    for case in cases:
        for query in selected_by_case.get(case.case_id, []):
            selected_items.append((case, query))

    score_cache = precompute_score_cache(
        text_embeddings.embeddings,
        image_embeddings.embeddings,
        [query.query_id for _case, query in selected_items],
        resolved_device,
        logger=logger,
    )

    attempted_pairs = 0
    valid_pairs = 0
    attempted_by_case = Counter()
    skip_counts = Counter()
    candidate_shifts: list[float] = []
    valid_shifts: list[float] = []
    start_time = time.time()
    last_log_time = start_time

    for item_idx, (case, query) in enumerate(selected_items, start=1):
        for trial_id in range(args.num_trials):
            attempted_pairs += 1
            attempted_by_case[case.case_id] += 1
            pair, reason = construct_gallery_pair(
                query_id=query.query_id,
                query_pid=query.pid,
                trial_id=trial_id,
                case_id=case.case_id,
                cue_a=case.cue_a,
                cue_b=case.cue_b,
                gallery_pids=split_data.gallery_pids,
                cue_scores=cue_scores,
                gallery_size=args.gallery_size,
                dense_ratio=args.dense_ratio,
                lambda_contrast=args.lambda_contrast,
                neutral_strategy=args.neutral_strategy,
                neutral_pool_factor=args.neutral_pool_factor,
                seed=args.seed,
            )
            if pair is None:
                skip_counts[reason] += 1
                tables["validity_counts"].append(
                    {
                        "dataset": args.dataset,
                        "split": args.split,
                        "retriever_name": args.retriever_name,
                        "case_id": case.case_id,
                        "query_id": query.query_id,
                        "trial_id": trial_id,
                        "attempted": 1,
                        "valid": 0,
                        "skip_reason": reason,
                        "candidate_cue_shift": "",
                        "cue_shift": "",
                    }
                )
                tables["skipped_queries"].append(
                    {
                        "dataset": args.dataset,
                        "split": args.split,
                        "retriever_name": args.retriever_name,
                        "case_id": case.case_id,
                        "query_id": query.query_id,
                        "trial_id": trial_id,
                        "reason": reason,
                        "details": {},
                    }
                )
                continue

            shift = cue_shift(
                pair.gallery_a,
                pair.gallery_b,
                query.pid,
                split_data.gallery_pids,
                cue_scores[case.cue_a],
                cue_scores[case.cue_b],
                cue_thresholds[case.cue_a],
                cue_thresholds[case.cue_b],
                args.tau_density,
            )
            candidate_shift = shift["cue_shift"]
            candidate_shifts.append(candidate_shift)
            if candidate_shift < args.min_pair_cue_shift:
                reason = "weak_cue_shift"
                skip_counts[reason] += 1
                tables["validity_counts"].append(
                    {
                        "dataset": args.dataset,
                        "split": args.split,
                        "retriever_name": args.retriever_name,
                        "case_id": case.case_id,
                        "query_id": query.query_id,
                        "trial_id": trial_id,
                        "attempted": 1,
                        "valid": 0,
                        "skip_reason": reason,
                        "candidate_cue_shift": candidate_shift,
                        "cue_shift": "",
                    }
                )
                tables["skipped_queries"].append(
                    {
                        "dataset": args.dataset,
                        "split": args.split,
                        "retriever_name": args.retriever_name,
                        "case_id": case.case_id,
                        "query_id": query.query_id,
                        "trial_id": trial_id,
                        "reason": reason,
                        "details": {"candidate_cue_shift": candidate_shift},
                    }
                )
                continue

            score_vector = score_cache[query.query_id]
            hm, reason = construct_hardness_controls(pair, query.pid, split_data.gallery_pids, score_vector)
            if hm is None:
                skip_counts[reason] += 1
                tables["validity_counts"].append(
                    {
                        "dataset": args.dataset,
                        "split": args.split,
                        "retriever_name": args.retriever_name,
                        "case_id": case.case_id,
                        "query_id": query.query_id,
                        "trial_id": trial_id,
                        "attempted": 1,
                        "valid": 0,
                        "skip_reason": reason,
                        "candidate_cue_shift": candidate_shift,
                        "cue_shift": "",
                    }
                )
                tables["skipped_queries"].append(
                    {
                        "dataset": args.dataset,
                        "split": args.split,
                        "retriever_name": args.retriever_name,
                        "case_id": case.case_id,
                        "query_id": query.query_id,
                        "trial_id": trial_id,
                        "reason": reason,
                        "details": {"candidate_cue_shift": candidate_shift},
                    }
                )
                continue

            ref_rank = retrieval_metrics(score_vector, split_data.gallery_pids, query.pid)["best_positive_rank"]
            hm_shift = cue_shift(
                hm.hm_a,
                hm.hm_b,
                query.pid,
                split_data.gallery_pids,
                cue_scores[case.cue_a],
                cue_scores[case.cue_b],
                cue_thresholds[case.cue_a],
                cue_thresholds[case.cue_b],
                args.tau_density,
            )

            a_row = _gallery_metrics_row(
                args,
                case,
                query,
                trial_id,
                GALLERY_TYPE_CUE_A,
                pair.gallery_a,
                score_vector,
                split_data,
                shift["cue_density_a_ga"],
                shift["cue_density_b_ga"],
                candidate_shift,
                ref_rank,
                score_mode,
            )
            b_row = _gallery_metrics_row(
                args,
                case,
                query,
                trial_id,
                GALLERY_TYPE_CUE_B,
                pair.gallery_b,
                score_vector,
                split_data,
                shift["cue_density_a_gb"],
                shift["cue_density_b_gb"],
                candidate_shift,
                ref_rank,
                score_mode,
            )
            hm_a_density_a = cue_density(
                hm.hm_a,
                query.pid,
                split_data.gallery_pids,
                cue_scores[case.cue_a],
                cue_thresholds[case.cue_a],
                args.tau_density,
            )
            hm_a_density_b = cue_density(
                hm.hm_a,
                query.pid,
                split_data.gallery_pids,
                cue_scores[case.cue_b],
                cue_thresholds[case.cue_b],
                args.tau_density,
            )
            hm_b_density_a = cue_density(
                hm.hm_b,
                query.pid,
                split_data.gallery_pids,
                cue_scores[case.cue_a],
                cue_thresholds[case.cue_a],
                args.tau_density,
            )
            hm_b_density_b = cue_density(
                hm.hm_b,
                query.pid,
                split_data.gallery_pids,
                cue_scores[case.cue_b],
                cue_thresholds[case.cue_b],
                args.tau_density,
            )
            hm_a_row = _gallery_metrics_row(
                args,
                case,
                query,
                trial_id,
                GALLERY_TYPE_HM_A,
                hm.hm_a,
                score_vector,
                split_data,
                hm_a_density_a,
                hm_a_density_b,
                hm_shift["cue_shift"],
                ref_rank,
                score_mode,
            )
            hm_b_row = _gallery_metrics_row(
                args,
                case,
                query,
                trial_id,
                GALLERY_TYPE_HM_B,
                hm.hm_b,
                score_vector,
                split_data,
                hm_b_density_a,
                hm_b_density_b,
                hm_shift["cue_shift"],
                ref_rank,
                score_mode,
            )

            cue_row, hm_row, delta_row = _paired_rows(
                args,
                case,
                query,
                trial_id,
                pair,
                a_row,
                b_row,
                hm_a_row,
                hm_b_row,
                hm,
                candidate_shift,
            )
            tables["per_gallery_results"].extend([a_row, b_row, hm_a_row, hm_b_row])
            tables["paired_cue_swap_results"].append(cue_row)
            tables["paired_hardness_control_results"].append(hm_row)
            tables["paired_delta_results"].append(delta_row)
            if args.save_galleries:
                tables["galleries"].extend(
                    _gallery_json_rows(
                        args,
                        case,
                        query,
                        trial_id,
                        {
                            GALLERY_TYPE_CUE_A: pair.gallery_a,
                            GALLERY_TYPE_CUE_B: pair.gallery_b,
                            GALLERY_TYPE_HM_A: hm.hm_a,
                            GALLERY_TYPE_HM_B: hm.hm_b,
                        },
                        split_data,
                        args.save_image_paths,
                    )
                )
            valid_pairs += 1
            valid_shifts.append(candidate_shift)
            tables["validity_counts"].append(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "retriever_name": args.retriever_name,
                    "case_id": case.case_id,
                    "query_id": query.query_id,
                    "trial_id": trial_id,
                    "attempted": 1,
                    "valid": 1,
                    "skip_reason": "",
                    "candidate_cue_shift": candidate_shift,
                    "cue_shift": candidate_shift,
                }
            )

        if item_idx % 50 == 0 or item_idx == len(selected_items):
            now = time.time()
            elapsed = now - start_time
            interval = now - last_log_time
            qps = 50.0 / interval if item_idx % 50 == 0 and interval > 0 else 0.0
            logger.info(
                "selected_queries %d/%d attempted_pairs=%d valid_pairs=%d "
                "score_cache_queries=%d elapsed=%.1fs interval=%.1fs qps=%.3f "
                "skip_counts=%s candidate Cue Shift stats=%s",
                item_idx,
                len(selected_items),
                attempted_pairs,
                valid_pairs,
                len(score_cache),
                elapsed,
                interval,
                qps,
                dict(skip_counts),
                _candidate_stats(candidate_shifts),
            )
            last_log_time = now

    logger.info("Final skip counts: %s", dict(skip_counts))
    logger.info("Final candidate Cue Shift stats: %s", _candidate_stats(candidate_shifts))
    logger.info("Final valid Cue Shift stats: %s", _candidate_stats(valid_shifts))

    bootstrap_results = _run_bootstraps(tables["paired_delta_results"], args, logger)
    primary_unit = _primary_bootstrap_unit(args.bootstrap_unit)
    tables["summary_with_ci"] = bootstrap_results[primary_unit]

    tables["summary_overall"], tables["summary_by_case"] = _summary_rows(
        args,
        whole_row,
        cases,
        selected_by_case,
        tables["paired_delta_results"],
        attempted_by_case,
        attempted_pairs,
    )

    row_counts = write_all_outputs(args.output_dir, tables)
    for unit, rows in bootstrap_results.items():
        filename = _bootstrap_filename(unit)
        row_counts[filename] = write_csv_rows(args.output_dir, filename, rows)
    logger.info("Output row counts: %s", row_counts)
    _log_final_report(logger, args.output_dir, tables["summary_overall"], tables["summary_with_ci"])
    valid_rate = valid_pairs / attempted_pairs if attempted_pairs else 0.0
    mean_shift = float(np.mean(valid_shifts)) if valid_shifts else 0.0
    audit_outputs(logger, row_counts, valid_rate, mean_shift, args.min_pair_cue_shift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
