# Phase B0 prediction-only qualitative painter

## Question

Can the completed cloud-native Phase B0 prediction-only model, which achieved
97.61% diagnostic four-way retrieval, use its learned one-stroke latent dynamics
to support a recognizable multi-step painting?

This command provides an inference-only qualitative diagnostic. It does not
repeat training, reopen the completed development decision, or constitute
formal Phase B0/B1/B2 evaluation.

## Exact checkpoint

The command accepts only the completed prediction-only checkpoint:

```text
checkpoints/joint_prediction_only_seed73.pt
artifact SHA-256: d13124a1becb7a11a84eb973a5d3acf72780f9813de6ed9422432778406155b4
state SHA-256: c1402cb94faadb504487d7be5295ae1c95a5f0b41dd8485768c9e20c5ed9e462
variant: joint_prediction_only
best epoch: 40
```

It verifies the entire checkpoint file, checkpoint metadata, architecture,
state-dict digest, and frozen parameter state before inference. The progress
model is deliberately rejected.

## Behavior

For one arbitrary input image, the command:

1. applies the existing deterministic 64x64 grayscale preprocessing and polarity repair;
2. starts from a white canvas;
3. proposes target-guided candidate strokes using the existing proposal policy;
4. asks the prediction-only model for each candidate's predicted next latent state;
5. selects the candidate predicted to be closest to the encoded target;
6. executes the selected stroke with the exact renderer and observes the real canvas again;
7. saves the best and final states;
8. runs an exact-pixel qualitative comparator under the same proposal settings;
9. replays both selected sequences at presentation resolution.

The exact comparator and per-step exact candidate ranks are diagnostics. They do
not affect the latent model's choices.

## Scientific boundary

- Source Phase B0 decision remains `not_eligible`.
- The consumed cloud-native training authorization is not reused.
- No model is trained, fine-tuned, or selected.
- No formal, B1, or B2 phase is authorized.
- The user image is exploratory and cannot alter historical results.
- The checkpoint and target are verified unchanged after inference.
- A successful-looking image would be qualitative evidence only.

## Local validation

```bash
git pull --ff-only
source .venv/bin/activate
python -m pytest -q
```

Expected total: `171 passed`.

## Recommended execution environment

Use a fresh Google Colab GPU runtime because the verified checkpoint already
resides in Drive and 100 rounds of 128-way convolutional scoring are faster on
the T4. The checkpoint path in the completed Drive output is:

```text
/content/drive/MyDrive/latent-stroke-dynamics-phase-b0-cloud-native/phase-b0-cloud-native-development-2026-08-24/checkpoints/joint_prediction_only_seed73.pt
```

The command is:

```bash
python paint_phase_b0_latent.py \
  --target "/path/to/target.png" \
  --checkpoint "/content/drive/MyDrive/latent-stroke-dynamics-phase-b0-cloud-native/phase-b0-cloud-native-development-2026-08-24/checkpoints/joint_prediction_only_seed73.pt" \
  --output-dir "/content/drive/MyDrive/latent-stroke-dynamics-phase-b0-qualitative/mnist-3-prediction-only" \
  --polarity auto \
  --steps 100 \
  --candidates 128 \
  --seed 20261001 \
  --prediction-batch-size 32 \
  --device cuda:0 \
  --high-res-size 512 \
  --supersample 2
```

Use a new output directory for every target. Completed and `.incomplete`
directories are never overwritten.

## Main artifacts

```text
target_64.png
target_512.png
summary.json
run_config.json
comparison.png
progress_comparison.png
latent_prediction_only/best_64.png
latent_prediction_only/final_64.png
latent_prediction_only/best_512.png
latent_prediction_only/final_512.png
latent_prediction_only/painting_512.gif
latent_prediction_only/steps.csv
exact_pixel/best_512.png
exact_pixel/final_512.png
exact_pixel/painting_512.gif
```
