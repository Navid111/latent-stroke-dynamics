# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/planner-score-development-results.md`;
3. `docs/planner-score-audit-results.md`;
4. `docs/planner-score-alignment-protocol.md`;
5. `docs/latent-planner-controlled-results.md`;
6. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every formal one-step criterion passed.

The controlled multi-step latent planner completed with implementation integrity passed but failed one required criterion. Latent ranking improved all six targets and beat random by 38.10%, but its mean final MSE was 1.996× exact pixel, above the frozen 1.5× maximum.

The Stage A score audit selected MSE-only plus normalized-latent L1. In long-horizon development, forced L1 reduced mean final MSE by about 7.05% versus forced latent MSE and was lower on all three development targets. The zero-margin no-op stopped after only 3.33 strokes on average and failed both scientific eligibility criteria.

## Active task

Archive the completed Stage A evidence and write the thesis. The planner-development result is `not_eligible`; confirmatory evaluation is unauthorized and must not run.

## Hard boundaries

- Do not rerun or tune the completed score audit, planner development, smoke, or controlled comparison.
- Do not train or fine-tune against any completed target.
- Do not change or overwrite any completed result.
- Keep the selected pair fixed as MSE-only plus normalized-latent L1.
- Do not tune the no-op margin on development targets.
- Do not run the reserved confirmatory phase.
- Preserve positive and negative outcomes.
