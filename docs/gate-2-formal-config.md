# Frozen formal Gate 2 configuration

**Status:** Frozen  
**Frozen on:** 2026-08-21  
**Formal data viewed:** No  
**Prerequisite evidence:** `docs/gate-2-dev-v2.md` and `docs/gate-2-retrieval-diagnostics.md`

## Rationale

Development v2 established a stable M1 execution path, converged validation curves, strong average-error prediction, unique counterfactual outcomes, and a specific width-discrimination weakness rather than an implementation failure. Hyperparameters are therefore frozen before generating or viewing any amended formal split.

No scientific setting may be changed after the formal run begins. A retry with `--reuse-cache` is permitted only after an interrupted or failed process that did not produce a completed result; a completed formal result must not be rerun for selection.

## Exact command

Run from the repository root with the existing virtual environment active:

```bash
python experiments/02_one_step_prediction.py \
  --model facebook/dinov2-small \
  --canvas-size 64 \
  --crowding 0 5 15 \
  --train-samples 1000 \
  --val-samples 200 \
  --test-samples 300 \
  --stress-samples 100 \
  --train-seed 20260824 \
  --val-seed 20260825 \
  --test-seed 20260826 \
  --stress-seed 20260827 \
  --model-seeds 11 22 33 \
  --epochs 30 \
  --patience 6 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --hidden-dim 256 \
  --overfit-examples 4 \
  --overfit-steps 30 \
  --overfit-learning-rate 0.005 \
  --encode-batch-size 8 \
  --encode-chunk-size 32 \
  --train-batch-size 16 \
  --encoder-device cpu \
  --train-device cpu \
  --formal-run \
  --output-dir outputs/gate2-formal-2026-08-21
```

## Frozen model-selection rule

Train both linear and MLP families with seeds `11`, `22`, and `33`. Select the family with the lowest mean validation action-region MSE across its three seeds. Apply the already-frozen Gate 2 decision rule to that family averaged across its three formal seeds.

## Expected eligibility

`run_config.json` must show:

```text
formal_run_requested: true
formal_eligible: true
```

If `formal_eligible` is false, do not interpret the output as the formal result. Preserve the output and diagnose only the configuration mismatch.

## Resource expectation

The base-model M1 completed 256/64/96 development encoding comfortably. The formal run is larger and trains six learned models, so it may take substantially longer. Feature caches may approach roughly one gigabyte. Keep the machine connected to power, prevent sleep, and allow the command to finish without opening memory-heavy applications.
