# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/ranking-aware-latent-formal-results.md`;
3. prior frozen results and protocols;
4. relevant source files and tests.

## Frozen evidence

The single formal ranking-aware comparison succeeded and is immutable. Do not rerun, retune, regenerate, or overwrite it.

Primary formal result: MSE-only retrieval 31.44%; ranking-aware retrieval 74.44%; gain 43.00 points; all preregistered criteria and integrity checks passed.

## Active task

Design and freeze a separate latent-space candidate-selection painter protocol before implementation. It may use the saved formal ranking-aware checkpoints but must not retrain or reinterpret the formal experiment.

## Hard boundaries

- Never rerun formal experiment 14, development experiment 12, or representation experiment 10.
- Preserve the completed formal output and every raw/adjudicated artifact.
- Do not use formal test or stress performance to select a deployment checkpoint.
- No encoder/predictor retraining for the first latent-planner study.
- Do not call this a canonical JEPA.
