# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/phase-b-saliency-latent-protocol.md`;
3. `docs/planner-score-development-results.md`;
4. `docs/planner-score-audit-results.md`;
5. `docs/planner-score-alignment-protocol.md`;
6. `docs/latent-planner-controlled-results.md`;
7. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every formal one-step criterion passed.

The controlled multi-step latent planner completed with implementation integrity passed but failed one required criterion. Latent ranking improved all six targets and beat random by 38.10%, but its mean final MSE was 1.996× exact pixel, above the frozen 1.5× maximum.

The Stage A score audit selected MSE-only plus normalized-latent L1. In long-horizon development, forced L1 reduced mean final MSE by about 7.05% versus forced latent MSE and was lower on all three development targets. The zero-margin no-op stopped after only 3.33 strokes on average and failed both scientific eligibility criteria.

## Active task

Phase B0 has a new frozen protocol on branch `phase-b/saliency-latent`. Implement only the configuration validator, fixed architecture, objective utilities, and validation-only tests. Phase B0 development data, formal B0, saliency scheduling B1, and RGB/high-resolution B2 are all unauthorized.

## Phase B0 scientific purpose

Test a multi-scale action-conditioned joint-embedding model with rendered spatial action conditioning and a calibrated exact-progress head. B0 remains 64×64 grayscale with the existing renderer so model and objective effects are isolated. Saliency scheduling and color are later, separately frozen phases.

## Hard boundaries

- Do not rerun or tune any completed experiment.
- Do not train or fine-tune against any completed target.
- Do not change or overwrite any completed result.
- Do not tune the Stage A no-op margin or run its reserved confirmatory phase.
- Do not generate any Phase B renderer transition, target, state bank, candidate set, checkpoint, or output before a separate authorization commit.
- Validation may use deterministic random dummy tensors only and may not load historical checkpoints.
- Do not alter the frozen Phase B0 architecture, objectives, seeds, thresholds, or compute cap after implementation begins.
- Preserve positive and negative outcomes.
- Do not call the approach a canonical JEPA.
