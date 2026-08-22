# Stage 3 learned-painter checkpoint handoff — 2026-08-22

## Implemented

- strict demonstration-checkpoint metadata;
- atomic model saving and state-dict integrity digest;
- safe checkpoint loading and architecture reconstruction;
- train/validation-only checkpoint trainer;
- memory-batched candidate prediction;
- learned candidate ranking followed by exact execution;
- exact rank, top-1, top-5, regret, and true-improvement diagnostics;
- three new tests, including an exact-oracle end-to-end learned-loop check.

## Local validation

```bash
git pull
source .venv/bin/activate
pytest
```

Expected collection: 33 tests.

## Train the demonstration checkpoint

Only after all tests pass:

```bash
python experiments/05_train_pixel_planner_checkpoint.py
```

The script generates only the fixed 1,000 train and 200 validation rows. It does not generate, evaluate, or select on paired test rows.

Expected local files:

```text
checkpoints/stage3-pixel-mlp-seed11.pt
outputs/stage3-demo-checkpoint/training_history.csv
outputs/stage3-demo-checkpoint/checkpoint_metadata.json
outputs/stage3-demo-checkpoint/training_summary.json
```

`checkpoints/` is ignored by Git and the weights must remain local.

## What to send back

1. Full pytest result.
2. Complete checkpoint-training terminal output.
3. `training_summary.json`.
4. `checkpoint_metadata.json`.
5. `training_history.csv` only if the terminal or summary indicates an issue.

After review, the next commit will add the all-method smoke and user-facing output path using the saved checkpoint.
