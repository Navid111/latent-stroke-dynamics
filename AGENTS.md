# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. frozen core and representation-extension results;
3. `docs/ranking-aware-latent-protocol.md`;
4. `docs/ranking-aware-latent-development-results.md`;
5. relevant source files and tests.

## Frozen evidence

All prior experiments are complete and immutable. The ranking-aware development grid also completed once and must not be rerun.

## Active task

Validate the no-rerun development adjudicator:

```bash
pytest
python experiments/13_ranking_development_adjudication.py
```

It reads only saved JSON/CSV artifacts. It loads no model/data, trains nothing, generates no data, and recomputes no scientific metric.

## Known reporting issue

The raw integrity failure came from structural NaNs created when heterogeneous MSE-only and ranking-aware history rows were concatenated. Method-applicable loss columns must be checked separately. Preserve the raw summary unchanged.

## Hard boundaries

- Never rerun the development grid.
- Do not generate formal seeds `20261104`–`20261110`.
- Do not freeze the selected formal setting until adjudication validates.
- Do not run experiment 10 again or retrain the encoder.
- Do not change canvas, renderer, strokes, architecture, grid, or thresholds.
