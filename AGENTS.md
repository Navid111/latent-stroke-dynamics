# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. frozen core and representation-extension results;
3. `docs/ranking-aware-latent-protocol.md`;
4. `docs/ranking-aware-latent-development-results.md`;
5. `docs/ranking-aware-latent-formal-handoff.md`;
6. relevant source files and tests.

## Frozen evidence

All prior experiments and the ranking-aware development grid are complete and immutable.

## Active task

One formal ranking-aware comparison is authorized after successful local validation:

```bash
pytest
python experiments/14_ranking_aware_latent_formal.py --validate-only
python experiments/14_ranking_aware_latent_formal.py --run-formal-comparison
```

## Hard boundaries

- Run the formal comparison exactly once.
- If it fails, preserve `.incomplete` and report the traceback; do not retry.
- If it succeeds, do not rerun or retune it.
- Never rerun development or experiment 10.
- Do not change lambda `1.0`, temperature `0.05`, encoder, canvas, renderer, strokes, architecture, thresholds, split sizes, or seeds.
- Test/stress rows cannot affect training, early stopping, or selection.
- Preserve all raw and adjudicated artifacts.
