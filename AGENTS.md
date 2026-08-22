# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. frozen core and representation-extension results;
3. `docs/ranking-aware-latent-protocol.md`;
4. `docs/ranking-aware-latent-handoff.md`;
5. relevant source files and tests.

## Frozen evidence

All Gate 1, Gate 2, pixel-control, Stage 3, qualitative, and representation-extension outcomes are complete and immutable.

## Active study

Exactly one ranking-aware development-grid execution is authorized after 66 passing tests and successful no-data validation.

Run exactly:

```bash
python experiments/12_ranking_aware_latent_followup.py --run-development-grid
```

Development may select only ranking lambda and temperature using validation metrics. Formal data remain unauthorized.

## Hard boundaries

- Execute the development grid once only.
- Do not add flags or run a concurrent second process.
- On failure, preserve the traceback and `.incomplete` directory; do not retry without review.
- Do not generate formal seeds `20261104`–`20261110`.
- Do not run experiment 10 again.
- Do not retrain/fine-tune the frozen task autoencoder.
- Do not change canvas, renderer, strokes, architecture, grid, thresholds, or split sizes.
- Do not use diagnostic-test values for ranking-setting selection.

## Code standards

Preserve atomic outputs and every negative outcome. Never commit credentials, checkpoints, caches, generated datasets, or raw output directories.
