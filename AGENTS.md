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

Validate the implemented formal ranking-aware comparison without data generation:

```bash
pytest
python experiments/14_ranking_aware_latent_formal.py --validate-only
```

Formal execution remains unauthorized until local validation is reviewed and a separate authorization commit lands.

## Hard boundaries

- Do not use `--run-formal-comparison` yet.
- Never rerun development or experiment 10.
- Do not generate formal seeds before authorization.
- Do not change lambda `1.0`, temperature `0.05`, encoder, canvas, renderer, strokes, architecture, thresholds, split sizes, or seeds.
- Test/stress rows cannot affect training, early stopping, or selection.
- Preserve all raw and adjudicated development artifacts.
