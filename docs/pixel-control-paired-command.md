# Frozen paired pixel-control command

**Status:** Authorized after smoke review  
**Configuration source:** `configs/pixel-control-paired-2026-08-21.json`  
**Run count:** Once

Run after pulling the latest documentation commit:

```bash
python experiments/03_pixel_space_control.py \
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
  --hidden-dim 64 \
  --overfit-examples 4 \
  --overfit-steps 30 \
  --overfit-learning-rate 0.005 \
  --train-batch-size 16 \
  --train-device cpu \
  --paired-control-run \
  --output-dir outputs/pixel-control-paired-2026-08-21
```

Expected eligibility labels are:

```text
paired_control_eligible: True
control_status: success | partial | failure | invalid
```

Do not interrupt a healthy run. Do not rerun, change seeds, change epochs, or tune after seeing the paired output. A plotting warning alone does not justify rerunning. The control result is explanatory and cannot revise the recorded latent Gate 2 fail.

After completion, provide the full terminal output and the first artifact batch:

1. `run_config.json`;
2. `control_diagnostics.csv`;
3. `retrieval_family_summary.csv`;
4. `aggregate_metrics.csv`;
5. `aggregate_metrics_by_crowding.csv`.
