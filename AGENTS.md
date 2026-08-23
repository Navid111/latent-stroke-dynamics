# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/latent-planner-protocol.md`;
3. `docs/ranking-aware-latent-formal-results.md`;
4. relevant planning, encoding, checkpoint, and test code.

## Frozen evidence

The single formal ranking-aware comparison succeeded and is immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every criterion passed.

The latent-planner foundation passed with 79 tests. Six formal latent predictor state hashes are frozen in the planner config. The guarded smoke runner then passed all 84 tests and its no-data unauthorized validation.

## Active task

Execute the one authorized five-method latent-planner smoke exactly once, preserve all outputs, and review implementation integrity. The controlled comparison remains unauthorized.

## Hard boundaries

- Do not rerun the smoke after a completed or `.incomplete` output appears.
- Do not generate any controlled targets.
- Do not train or fine-tune any model.
- Do not rerun formal experiment 14, development experiment 12, or representation experiment 10.
- Do not change scoring, ensemble, proposal, target, step, candidate, threshold, or seed settings.
- Do not use formal test/stress results for deployment checkpoint selection.
- Preserve all completed raw and adjudicated artifacts.
