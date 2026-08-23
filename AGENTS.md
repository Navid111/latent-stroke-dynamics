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

The latent-planner foundation passed with 79 tests. Six formal latent predictor state hashes are frozen in the planner config. No planner data were generated and no model was trained or fine-tuned.

## Active task

Implement and validate a guarded all-five-method latent-planner smoke runner. Validation must exercise only the unauthorized path and must not generate the reserved target or create output directories. A separate authorization commit is required before the one smoke execution.

## Hard boundaries

- Do not generate latent-planner smoke or controlled targets yet.
- Do not train or fine-tune any model.
- Do not rerun formal experiment 14, development experiment 12, or representation experiment 10.
- Do not change scoring, ensemble, proposal, target, step, candidate, threshold, or seed settings.
- Do not use formal test/stress results for deployment checkpoint selection.
- Preserve all completed raw and adjudicated artifacts.
