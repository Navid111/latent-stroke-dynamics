# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. frozen core and representation-extension results;
3. `docs/ranking-aware-latent-protocol.md`;
4. `docs/ranking-aware-latent-development-results.md`;
5. `configs/ranking-aware-latent-selected-setting-2026-08-22.json`;
6. relevant source files and tests.

## Frozen evidence

All prior experiments and the ranking-aware development grid are complete and immutable. Development must never be rerun.

## Active task

Implement a guarded formal runner using only:

- ranking weight `1.0`;
- temperature `0.05`;
- untouched formal seeds `20261104`–`20261110`;
- the frozen task autoencoder/statistics;
- three matched MSE-only and three ranking-aware MLP seeds.

Formal execution is not yet authorized. Implement validation-only mode and tests first.

## Hard boundaries

- Never rerun development or experiment 10.
- Do not generate formal data before separate validation and authorization.
- Do not change the selected setting, encoder, canvas, renderer, strokes, architecture, thresholds, split sizes, or seeds.
- Primary test and stress data cannot affect training, early stopping, or selection.
- Preserve raw and adjudicated development outcomes.
