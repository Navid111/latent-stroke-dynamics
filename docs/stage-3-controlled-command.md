# Frozen Stage 3 controlled command — 2026-08-22

## Frozen assets

- config: `configs/stage3-controlled-2026-08-22.json`;
- config commit: `c0f0da9138bbbcd9a4952281ec21faf2f7bd9f56`;
- runner: `experiments/07_pixel_planner_controlled.py`;
- runner commit: `132992362cc74c9af91538f4de8c623c6b928348`;
- checkpoint digest: `e32f3612f7a184e4e9b58f95a987551bd25cdb17ff1bf2b6be40fcf5781ea472`.

## Frozen comparison

- six independently seeded 20-stroke synthetic targets;
- random, exact greedy, and learned pixel planners;
- 100 executed strokes per method and target;
- 128 candidates per step;
- 32-candidate learned inference batches;
- exact execution after every selected stroke;
- deterministic learned replay for every target;
- success criteria committed before controlled outputs exist.

## Step 1 — validation only

```bash
git pull
source .venv/bin/activate
python experiments/07_pixel_planner_controlled.py --validate-only
```

This checks the frozen config, local checkpoint, metadata, digest, and absence of prior controlled output. It does not generate target results.

Expected final line:

```text
No controlled data were generated.
```

## Step 2 — single controlled run

Run this only after validation succeeds:

```bash
python experiments/07_pixel_planner_controlled.py --controlled-run
```

The runner refuses to overwrite a completed result. It writes to an `.incomplete` directory and renames it to the final output only after all six targets, integrity checks, aggregation, figures, and decision logic complete.

## Success criteria

The learned planner succeeds only if:

1. final MSE improves from white on all six targets;
2. mean final MSE is at least 20% below random;
3. mean final MSE is no more than 25% above exact greedy;
4. implementation integrity passes.

These criteria are conjunctive and cannot be revised after the run.

## Expected final files

```text
outputs/stage3-controlled-2026-08-22/
├── aggregate_summary.csv
├── per_target_summary.csv
├── learned_step_diagnostics.csv
├── progress_by_step.csv
├── decision.csv
├── decision.json
├── run_config.json
├── aggregate_progress.png
├── final_montage.png
└── targets/
```

Each target directory also contains method-specific progress, stroke JSON, final images, GIFs, and comparison figures.

## After completion

Do not rerun or retune regardless of pass or fail. Send the terminal decision plus the aggregate summary, decision JSON, per-target summary, aggregate progress figure, and final montage for review.
