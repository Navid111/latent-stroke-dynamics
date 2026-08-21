# Pixel-space control handoff

The protocol was committed before implementation. The new experiment contains no frozen encoder and should be substantially lighter than latent Gate 2.

## Step 1 — update and test

```bash
git pull
pytest
```

The complete suite should collect 24 tests. Do not continue if a test fails.

## Step 2 — development-only engineering smoke

```bash
python experiments/03_pixel_space_control.py \
  --train-samples 128 \
  --val-samples 32 \
  --test-samples 64 \
  --stress-samples 0 \
  --train-seed 20260830 \
  --val-seed 20260831 \
  --test-seed 20260832 \
  --stress-seed 20260833 \
  --model-seeds 11 \
  --epochs 12 \
  --patience 4 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --hidden-dim 64 \
  --overfit-examples 4 \
  --overfit-steps 30 \
  --overfit-learning-rate 0.005 \
  --train-batch-size 16 \
  --train-device cpu \
  --output-dir outputs/pixel-control-smoke-1
```

This run must report `paired_control_eligible: false` and `control_status: diagnostic_only`. Those labels are expected; smoke data cannot decide the control result.

## Send for review

Paste the complete terminal output and send:

1. `control_diagnostics.csv`;
2. `retrieval_family_summary.csv`;
3. `aggregate_metrics.csv`;
4. `candidate_selection_distribution.png`;
5. `pairwise_true_win_rates.png`.

Keep the remaining files locally. Do not run the paired control command until the engineering smoke is reviewed.

## Frozen paired command

The paired command is recorded in `configs/pixel-control-paired-2026-08-21.json`. It must not be run until smoke review confirms implementation integrity. The completed latent Gate 2 result remains unchanged regardless of the pixel outcome.
