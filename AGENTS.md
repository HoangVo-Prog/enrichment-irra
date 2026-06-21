# AGENTS.md - IRRA Cue-Swap Diagnostic Port

## Project Goal

This IRRA repository should receive a faithful reimplementation of the existing
`diagnostic/*` cue-swap diagnostic package from the prototype repository.

The purpose is to evaluate a frozen IRRA text-based person search retriever
under matched cue-biased gallery perturbations, using an independent
off-the-shelf CLIP cue scorer for gallery construction and Cue Shift validity.

This diagnostic is a matched empirical probe, not a universal theorem about all
TBPS retrievers. It tests whether a representative frozen gallery-agnostic TBPS
retriever exhibits retrieval sensitivity when the distribution of visual cues
among unlabeled distractors is perturbed. The hardness-matched control tests
whether the effect exceeds what can be explained by replacing distractors with
similarly hard negatives.

## First Step: Inspect The IRRA Source Code

Before writing code, inspect the IRRA repository and adapt to its actual APIs.
Do not assume the prototype repo API exists.

Required inspection commands or equivalent:

```bash
rg -n "build_model|class .*IRRA|encode_text|encode_image|forward|checkpoint|Checkpointer|load_state_dict|Evaluator|rank\\(|TextDataset|ImageDataset|build_dataloader|tokenize|SimpleTokenizer|CLIP|clip" -S .
rg --files
```

Read the relevant files completely enough to answer:

1. How are datasets constructed for `CUHK-PEDES`, `ICFG-PEDES`, and `RSTPReid`?
2. What objects contain gallery image paths, gallery pids, captions, and caption pids?
3. What transform and tokenizer should evaluation use?
4. How is the IRRA model built from config?
5. How is a checkpoint loaded?
6. Which method returns text embeddings for retrieval?
7. Which method returns image embeddings for retrieval?
8. Does IRRA expose only one global retrieval branch, or multiple branches?
9. How does the official evaluation compute R@1, R@5, R@10, mAP, and ranks?
10. Is an off-the-shelf CLIP loader already present? If yes, reuse it without loading IRRA fine-tuned weights. If no, use the standard `clip`/OpenCLIP package already supported by the repo environment.

Only after this inspection should implementation begin.

## Non-Negotiable Scientific Constraints

1. Do not train or fine-tune any model.
2. Do not update IRRA weights.
3. Do not modify the training pipeline.
4. Do not use IRRA to construct cue-biased galleries.
5. Use an independent off-the-shelf CLIP cue scorer for cue affinity.
6. Do not call CLIP cue affinities ground-truth labels.
7. Identity labels may be used only to select positives, exclude same-identity distractors, and compute retrieval metrics.
8. Cue Shift must be computed over distractors only, not positives.
9. Hardness matching must use IRRA retrieval scores, not CLIP cue scores.
10. Bootstrap confidence intervals must resample query clusters, not individual trials.
11. Keep all sampling deterministic under fixed seed.
12. Write stable CSV/JSON schemas even when no valid rows exist.

## Required Folder Layout

Create the full package below. Reimplement all modules; do not leave monolithic logic in `tools/*`.

```text
diagnostic/
  __init__.py
  README.md
  run_cue_swap_diagnostic.py
  aggregate_diagnostic_tables.py
  plot_positive_ratio_audit.py
  config.py
  constants.py
  cue_ontology.py
  cue_cases.py
  data_loading.py
  retriever_loading.py
  clip_cue_scorer.py
  embeddings.py
  scoring.py
  gallery_construction.py
  controls.py
  metrics.py
  bootstrap.py
  outputs.py
  audit.py
```

## Required CLI

The main entrypoint must be:

```bash
python diagnostic/run_cue_swap_diagnostic.py
```

It must support:

```text
--dataset {CUHK-PEDES,ICFG-PEDES,RSTPReid}
--split {test,val}
--retriever_name irra
--retriever_config PATH
--retriever_checkpoint PATH
--cue_scorer off_the_shelf_clip
--clip_model_name TEXT
--output_dir PATH
--root_dir PATH optional
--cases_file PATH optional
--auto_cases optional flag
--cue_vocab_file PATH optional
--gallery_size INT
--dense_ratio FLOAT
--num_trials INT
--seed INT
--device TEXT
--score_mode {auto,global}
--lambda_global FLOAT accepted for schema compatibility, but ignored unless IRRA has multiple score branches
--lambda_contrast FLOAT
--cue_threshold_quantile FLOAT
--tau_density FLOAT
--min_pair_cue_shift FLOAT
--bootstrap_iters INT
--bootstrap_seed INT
--enable_random_control optional flag, may raise NotImplementedError
--save_galleries optional flag
--save_image_paths optional flag
--max_queries_per_case INT optional
--min_queries_per_auto_case INT
--max_auto_cases INT optional
--test_batch_size INT optional
--num_workers INT optional
--neutral_strategy {low_affinity,random}
--neutral_pool_factor INT
--dry_run
```

If IRRA actually exposes multiple retrieval branches, add extra score modes only after inspecting and documenting those branches. Otherwise `auto` resolves to `global`.

## Module Responsibilities

### `diagnostic/config.py`

Implement:

- argparse
- config loading
- validation
- deterministic seed setup
- device resolution
- JSON-safe serialization helpers
- stable hash-based seed helper

Do not import heavyweight image or model packages for `--dry_run` unless necessary.

### `diagnostic/constants.py`

Define:

- dataset names
- retriever names: `("irra",)`
- cue scorer names: `("off_the_shelf_clip",)`
- gallery type names
- output filenames
- stable output schemas
- prompt templates for cue scoring

Keep output column names compatible with the prototype diagnostic package:

```text
config_used.json
whole_test_metrics.csv
cue_thresholds.csv
cue_case_candidates.csv
cue_case_constructibility.csv
selected_queries.csv
validity_counts.csv
per_gallery_results.csv
paired_cue_swap_results.csv
paired_hardness_control_results.csv
paired_delta_results.csv
summary_overall.csv
summary_by_case.csv
summary_with_ci.csv
skipped_queries.jsonl
galleries.jsonl
```

### `diagnostic/data_loading.py`

Reuse IRRA dataset classes and evaluation transforms when possible.

Return normalized records:

```python
QueryRecord(query_id: int, text: str, pid: int)
GalleryRecord(image_id: int, path: str, pid: int)
SplitData(
    dataset,
    query_records,
    gallery_records,
    query_pids,
    gallery_pids,
    gallery_paths,
    img_loader,
    txt_loader,
    num_classes,
)
```

Also implement metadata-only loading for `--dry_run` if possible, so cue-case validation can run without reading images or importing model code.

### `diagnostic/retriever_loading.py`

This is the most IRRA-specific module.

Implement an adapter:

```python
class RetrieverAdapter:
    name: str
    model: torch.nn.Module
    device: torch.device
    has_grab: bool = False

    def eval(self): ...
    def encode_text(self, batch): ...
    def encode_image(self, batch): ...
```

Adapt `encode_text` and `encode_image` to the actual IRRA model API discovered during inspection.

Common patterns to check:

- `model.encode_text(tokens)`
- `model.encode_image(images)`
- `model.base_model.encode_text(tokens)`
- `model.base_model.encode_image(images)`
- official evaluator embedding extraction helpers

Checkpoint loading must use the IRRA repository's official loader when available. If the checkpoint is a dict containing `"model"`, load that state dict. If it uses `"state_dict"` or another key, support it after inspection.

All parameters must be frozen:

```python
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
```

### `diagnostic/clip_cue_scorer.py`

Instantiate a separate off-the-shelf CLIP model for cue scoring.

Requirements:

- Do not load IRRA checkpoint weights.
- Do not use IRRA embeddings for cue construction.
- Support prompt ensemble.
- Normalize embeddings before cosine similarity.
- Export cue prompts and cue thresholds.

The scorer should expose:

```python
class OffTheShelfCLIPCueScorer:
    def encode_gallery_images(...): ...
    def encode_cues(cues, text_length): ...
    def score(cues, img_loader, text_length, logger): ...
```

If IRRA has a CLIP builder, use it only with original off-the-shelf weights. If the builder always loads fine-tuned IRRA state, do not use it for cue scoring.

### `diagnostic/embeddings.py`

Extract and cache IRRA query/gallery embeddings.

Use:

```python
torch.no_grad()
F.normalize(..., p=2, dim=1)
```

Do not use identity labels during inference.

### `diagnostic/scoring.py`

Implement:

- score mode resolution (`auto -> global`)
- query-gallery scoring
- selected full-score precomputation support
- whole-test sanity metrics
- deterministic rank sorting

For IRRA, global retrieval score is cosine similarity between normalized IRRA text and image embeddings.

### `diagnostic/gallery_construction.py`

Implement cue-swap gallery construction using external CLIP cue affinities:

```text
G_a = positives(q) + cue-a dense distractors + neutral fillers
G_b = positives(q) + cue-b dense distractors + neutral fillers
```

Rules:

- Same positives in both galleries.
- Same gallery size.
- No duplicate image ids.
- No same-identity distractors.
- Dense scores:

```text
score_a_dense = psi_a - lambda_contrast * psi_b
score_b_dense = psi_b - lambda_contrast * psi_a
```

- Neutral fillers should be low max cue-affinity or random, controlled by CLI.
- Deterministic with `seed, case_id, query_id, trial_id, gallery_type`.

### `diagnostic/controls.py`

Implement mandatory hardness-matched controls.

Hardness is:

```text
hardness(i | q, IRRA) = s_IRRA(q, i)
```

Use deterministic quantile-bin matching or another efficient score-distribution matching method. Do not use CLIP cue affinities here.

For each cue-swap pair, construct:

```text
hm_a = positives(q) + hardness-matched replacement for a_dense distractors
hm_b = positives(q) + hardness-matched replacement for b_dense distractors
```

Output diagnostics:

```text
hm_a_mean_score_cue_subset
hm_a_mean_score_control_subset
hm_a_std_score_cue_subset
hm_a_std_score_control_subset
hm_b_mean_score_cue_subset
hm_b_mean_score_control_subset
hm_b_std_score_cue_subset
hm_b_std_score_control_subset
```

### `diagnostic/metrics.py`

Implement:

```text
R1
R5
R10
AP
best_positive_rank
R@1 Flip
Rank Shift
AP Delta
Cue Shift
```

Cue Shift must use distractors only:

```text
N_q(G) = G \ positives(q)
D_c(G;q) = mean_{i in N_q(G)} sigmoid((psi_c(i) - eta_c) / tau_density)
CueShift = 0.5 * ((D_a(G_a;q) - D_a(G_b;q)) + (D_b(G_b;q) - D_b(G_a;q)))
```

### `diagnostic/bootstrap.py`

Implement cluster bootstrap over query clusters.

Cluster key:

```text
dataset, retriever_name, case_id, query_id
```

Do not bootstrap individual trials independently.

Use an optimized cluster-sum/count implementation, not a slow nested Pandas loop.

Metrics needing CIs:

```text
r1_flip
rank_shift
hm_r1_flip
hm_rank_shift
delta_r1_flip
delta_rank_shift
cue_shift
```

### `diagnostic/outputs.py`

Write all required CSV/JSONL outputs with stable schemas.

Do not silently skip files. Empty outputs must still have headers where applicable.

### `diagnostic/cue_ontology.py`

Reimplement the fixed cue ontology from the prototype diagnostic package:

- colored upper-body garments
- colored lower-body garments
- colored footwear
- bag/backpack/handbag/shoulder bag
- hat/cap/glasses/umbrella
- pattern cues

Support optional cue vocabulary JSON.

### `diagnostic/cue_cases.py`

Implement:

- manual cue-case loading
- query text filtering
- query id validation
- optional query regex
- automatic cue-case generation from query text and fixed ontology
- support filtering

Do not use retrieval metrics to choose or filter cases.

### `diagnostic/audit.py`

Implement warnings for:

- low valid-pair rate
- weak Cue Shift
- missing or empty output rows

### `diagnostic/aggregate_diagnostic_tables.py`

Read multiple diagnostic output folders and produce:

```text
generalized_cue_swap_table.csv
hardness_control_table.csv
validity_counts_table.csv
```

### `diagnostic/plot_positive_ratio_audit.py`

Read `per_gallery_results.csv` from one or more runs and write combined positive-ratio CSV/PNG audits.

## Performance Requirements

The implementation must include the current optimized behavior:

1. Precompute selected query-to-gallery score vectors in chunks when memory-safe.
2. Use CUDA for score precompute when `--device cuda`.
3. Cache full score vectors by `query_id`.
4. Use vectorized/quantile-bin hardness matching.
5. Use optimized cluster bootstrap over cluster sums/counts.
6. Log progress every 50 selected queries with:

```text
selected_queries current/total
attempted_pairs
valid_pairs
score_cache_queries
elapsed
interval
qps
skip_counts
candidate Cue Shift stats
```

The runner must also log:

- run parameters
- candidate/retained cue case counts
- selected query counts
- whole-test sanity metrics
- cue threshold stats
- final skip counts
- final candidate and valid Cue Shift stats
- bootstrap start/end timing
- output row counts

## Output Semantics

`summary_overall.csv` should include at least:

```text
dataset
retriever_name
cue_scorer
ref_R1
num_cases
num_queries
num_pairs
valid_pair_rate
mean_cue_shift
r1_flip
rank_shift
hm_r1_flip
hm_rank_shift
delta_r1_flip
delta_rank_shift
```

`summary_with_ci.csv` should include:

```text
metric
mean
ci_low
ci_high
bootstrap_iters
cluster_count
trial_count
```

## Example IRRA Run

Adjust paths to the actual IRRA repository:

```bash
python diagnostic/run_cue_swap_diagnostic.py \
  --dataset RSTPReid \
  --split test \
  --retriever_name irra \
  --retriever_config configs/rstpreid.yaml \
  --retriever_checkpoint /path/to/irra_best.pth \
  --cue_scorer off_the_shelf_clip \
  --clip_model_name ViT-B/16 \
  --output_dir outputs/diagnostic_rstp_irra \
  --gallery_size 500 \
  --dense_ratio 0.9 \
  --num_trials 3 \
  --score_mode global \
  --lambda_contrast 0.8 \
  --min_pair_cue_shift 0.0 \
  --max_queries_per_case 50 \
  --min_queries_per_auto_case 30 \
  --seed 42 \
  --device cuda
```

For fast debugging:

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
  --dry_run
```

Then run without `--dry_run` for a small real pass.

## Validation Checklist

Before finishing:

1. `python diagnostic/run_cue_swap_diagnostic.py --help` works.
2. `python diagnostic/aggregate_diagnostic_tables.py --help` works.
3. `python diagnostic/plot_positive_ratio_audit.py --help` works.
4. `python -m compileall diagnostic -q` passes.
5. Dry run validates cases and writes `config_used.json`.
6. One small real run completes on `RSTPReid`.
7. `whole_test_metrics.csv` has plausible IRRA R@1.
8. Cue scorer in `config_used.json` is explicitly `off_the_shelf_clip`.
9. `paired_cue_swap_results.csv` is non-empty for permissive `--min_pair_cue_shift 0.0`.
10. `paired_hardness_control_results.csv` is non-empty.
11. `paired_delta_results.csv` is non-empty.
12. `summary_with_ci.csv` contains CIs for all required metrics.
13. Logs show no long silent section.

## Claim-Safe Wording

Use:

```text
off-the-shelf CLIP cue scorer
external cue-affinity scores
cue-biased gallery perturbation
retrieval sensitivity
hardness-matched control
representative frozen gallery-agnostic retriever
```

Avoid:

```text
ground-truth cue labels
true attribute detector
proves all TBPS models are gallery-dependent
changes only cue distribution
```

## Final Deliverable

The final implementation in the IRRA source repository should be a complete
`diagnostic/` package with the same behavior, output files, schemas, logging,
and optimized algorithms as the prototype diagnostic package, adapted only where
necessary to IRRA's real dataset/model/checkpoint APIs.
