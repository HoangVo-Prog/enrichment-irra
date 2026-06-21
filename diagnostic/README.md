# IRRA Cue-Swap Diagnostic

This package evaluates a frozen IRRA text-based person search retriever under
matched cue-biased gallery perturbations. Gallery construction uses an
independent off-the-shelf CLIP cue scorer to produce external cue-affinity
scores; IRRA scores are used only for retrieval metrics and for the
hardness-matched control.

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
  --dry_run
```

On real runs, the retriever is built with `model.build_model`, checkpoint
loading follows the repository `Checkpointer`/`load_state_dict` helpers, and
retrieval uses the single global branch exposed by `model.encode_text` and
`model.encode_image`.

Generated files use stable schemas, including empty CSV headers when no valid
rows exist. Cue Shift is computed over distractors only.
