# IRRA Cue-Swap Diagnostic

This package evaluates a frozen IRRA text-based person search retriever under
matched cue-biased gallery perturbations. Gallery construction uses an
independent off-the-shelf CLIP cue scorer to produce external cue-affinity
scores; IRRA scores are used only for retrieval metrics and for the
hardness-matched control.

Cue-swap galleries share one neutral distractor set. For each query/trial, both
cue galleries contain the complete positive set and the same neutral
distractors; only the cue-dense distractor subset differs. The
hardness-matched control still replaces the full distractor set independently
for each cue direction.

For parity with the prototype diagnostic, the cue scorer uses the private
`diagnostic/prototype_clip_model.py` implementation instead of IRRA's
`model.clip_model`.

The main entrypoint is:

```bash
python diagnostic/run_cue_swap_diagnostic.py --help
```

Typical small dry run:

```bash
python diagnostic/run_cue_swap_diagnostic.py \
  --dataset RSTPReid \
  --split test \
  --retriever_name irra \
  --retriever_config configs/rstpreid.yaml \
  --retriever_checkpoint /path/to/irra_best.pth \
  --output_dir outputs/debug_irra \
  --gallery_size 100 \
  --num_trials 1 \
  --max_auto_cases 5 \
  --max_queries_per_case 5 \
  --bootstrap_iters 100 \
  --tight_hardness_z_tolerance 0.10 \
  --dry_run
```

On real runs, the retriever is built with `model.build_model`, checkpoint
loading follows the repository `Checkpointer`/`load_state_dict` helpers, and
retrieval uses the single global branch exposed by `model.encode_text` and
`model.encode_image`.

Generated files use stable schemas, including empty CSV headers when no valid
rows exist. Cue Shift is computed over distractors only. The run reports
hardest-negative and positive-negative margin audits in
`per_gallery_results.csv`, paired Cue-vs-HM hardness gaps in
`paired_delta_results.csv`, and summary files:

```text
hardness_audit_with_ci.csv
tight_hardness_summary_with_ci.csv
```

The HM control approximately matches retriever-score difficulty, and residual
top-rank mismatch is audited using hardest-negative scores, positive-negative
margins, and a fixed tight-match robustness subset controlled by
`--tight_hardness_z_tolerance` (default `0.10`). The tight-hardness subset is a
robustness result, not a replacement for the primary Cue-minus-HM result.

## Bootstrap Units

The diagnostic supports two cluster bootstrap units through:

```bash
--bootstrap_unit {case_query,unique_query,both}
```

`unique_query` is the recommended primary analysis.

Case-query-instance bootstrap:
Resamples each eligible `(case_id, query_id)` instance jointly with all of its
repeated trials. The cluster key is:

```text
dataset, retriever_name, case_id, query_id
```

Unique-query bootstrap:
Resamples each underlying `query_id` jointly with every associated cue case and
all repeated trials. The cluster key is:

```text
dataset, retriever_name, query_id
```

The reported metrics remain micro-averaged over valid case-query trials. The
unique-query bootstrap changes the uncertainty estimate, not the full-sample
point-estimate weighting. It is not a query-macro average.

The primary CI table is written to `summary_with_ci.csv`. Unit-specific tables
are written as requested:

```text
summary_with_ci_unique_query.csv
summary_with_ci_case_query.csv
```
