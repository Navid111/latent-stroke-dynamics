# Latent Stroke Dynamics

Bachelor's thesis experiments on **action-conditioned latent canvas dynamics for stroke-based rendering**.

## Current status

**Gate 1 passed on 2026-08-19.** The frozen DINOv2-small patch representation preserved controlled stroke changes under all primary crowding levels.

**Gate 2 formal configuration was frozen on 2026-08-21.** Development v2 showed strong average-error prediction but near-chance exact retrieval. A no-retraining decomposition identified stroke width as the main confusion rather than total action blindness. No implementation defect remained, so the architecture, loss, metrics, thresholds, hyperparameters, and untouched formal data are now frozen.

Key documents:

- [`docs/gate-2-dev-v2.md`](docs/gate-2-dev-v2.md)
- [`docs/gate-2-retrieval-diagnostics.md`](docs/gate-2-retrieval-diagnostics.md)
- [`docs/gate-2-formal-config.md`](docs/gate-2-formal-config.md)
- [`configs/gate2-formal-2026-08-21.json`](configs/gate2-formal-2026-08-21.json)

Gate 2 has not yet passed or failed.

## Formal Gate 2 run

Pull the frozen configuration:

```bash
git pull
```

Run the exact command once:

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

Keep the M1 connected to power, prevent sleep, and avoid memory-heavy applications. The run encodes six data splits and trains six learned models, so it will take substantially longer than development v2.

A successful configuration must report:

```text
formal_eligible: true
```

If the process is interrupted before completing, the identical command may be retried with `--reuse-cache`. Do not rerun a completed formal result.

## Frozen decision rule

The validation-selected family, averaged over seeds `11`, `22`, and `33`, must:

1. improve action-region MSE by at least 30% versus both identity and mean delta;
2. remain positive versus identity at crowding 0, 5, and 15;
3. achieve at least 50% counterfactual top-1 retrieval;
4. pass implementation and seed-stability checks.

Only the formal run may receive a Gate 2 decision. No target-guided planning, reinforcement learning, or multi-step rollout begins before that decision.
