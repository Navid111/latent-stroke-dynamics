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

The ranking-aware latent protocol, frozen input hashes, loss implementation, and guarded development-grid runner are committed. Development and formal data remain unauthorized.

Current action:

```bash
pytest
python experiments/12_ranking_aware_latent_followup.py --validate-only
```

Review local validation before changing either development authorization flag.

## Hard boundaries

- Do not use `--run-development-grid` yet.
- Do not create or modify the development output path during validation.
- Do not generate formal seeds `20261104`–`20261110`.
- Do not run experiment 10 again.
- Do not retrain/fine-tune the frozen task autoencoder.
- Do not change canvas, renderer, strokes, architecture, grid, thresholds, or split sizes.
- Do not use diagnostic-test values for ranking-setting selection.

## Code standards

Keep new code separate from completed experiments. Validate hashes before data generation. Use atomic outputs and refuse overwrite. Preserve every negative outcome. Never commit credentials, checkpoints, caches, generated datasets, or raw output directories.
