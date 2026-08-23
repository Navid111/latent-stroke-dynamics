# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/latent-planner-protocol.md`;
3. `docs/latent-planner-smoke-results.md`;
4. `docs/latent-planner-controlled-command.md`;
5. relevant planning, encoding, checkpoint, and test code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every criterion passed.

The latent-planner foundation passed and six predictor hashes were frozen. The one authorized smoke completed with full implementation integrity and is closed. The guarded controlled runner passed all 89 tests and no-data validation.

## Active task

Execute the one authorized six-target, five-method controlled comparison exactly once, preserve all outputs, and adjudicate only with the frozen criteria.

## Hard boundaries

- Do not rerun the completed smoke.
- Do not rerun the controlled comparison after a completed or `.incomplete` output appears.
- Do not train or fine-tune any model.
- Do not rerun any closed experiment.
- Do not change scoring, ensemble, proposal, target, step, candidate, threshold, or seed settings.
- Preserve all completed raw and adjudicated artifacts.
