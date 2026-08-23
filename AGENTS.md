# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/planner-score-alignment-protocol.md`;
3. `docs/latent-planner-controlled-results.md`;
4. `docs/latent-planner-protocol.md`;
5. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every formal one-step criterion passed.

The controlled multi-step latent planner completed with implementation integrity passed but failed one required criterion. Latent ranking improved all six targets and beat random by 38.10%, but its mean final MSE was 1.996× exact pixel, above the frozen 1.5× maximum. Latent MSE and learned pixel were stronger planners.

## Active task

Implement and validate the separately frozen Stage A planner-score alignment study. Stage A uses only new seeds and already-frozen models. Development score-audit data remain unauthorized until the validation-only runner passes review.

## Hard boundaries

- Do not rerun or retune the completed smoke or controlled comparison.
- Do not train or fine-tune against controlled targets.
- Do not change or overwrite any completed result.
- Do not rerun any closed experiment.
- Do not generate Stage A development, planner-development, or confirmatory data before its matching explicit authorization.
- Preserve positive and negative outcomes.
