# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/planner-score-audit-results.md`;
3. `docs/planner-score-development-command.md`;
4. `docs/planner-score-alignment-protocol.md`;
5. `docs/latent-planner-controlled-results.md`;
6. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every formal one-step criterion passed.

The controlled multi-step latent planner completed with implementation integrity passed but failed one required criterion. Latent ranking improved all six targets and beat random by 38.10%, but its mean final MSE was 1.996× exact pixel, above the frozen 1.5× maximum. Latent MSE and learned pixel were stronger planners.

The Stage A score audit completed with integrity passed. Its frozen development selection is the MSE-only ensemble with normalized-latent L1 scoring. Relative to MSE-only plus normalized-latent MSE, L1 reduced mean exact regret by about 28.8% and increased top-5 selection from 33.33% to 51.39%.

## Active task

Run the separately authorized three-target planner-development comparison exactly once. It compares exact pixel, learned pixel, current latent MSE forced horizon, selected latent L1 forced horizon, and selected latent L1 with the preregistered zero-margin no-op. Confirmatory evaluation remains unauthorized.

## Hard boundaries

- Do not rerun or tune the completed score audit, smoke, or controlled comparison.
- Do not train or fine-tune against any completed target.
- Do not change or overwrite any completed result.
- Keep the selected pair fixed as MSE-only plus normalized-latent L1.
- Run `--planner-development` at most once under the current authorization.
- Preserve an incomplete output directory after any error or interruption; do not retry without review.
- Do not generate confirmatory data before explicit authorization.
- Preserve positive and negative outcomes.
