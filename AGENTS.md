# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/phase-b-saliency-latent-protocol.md`;
3. `docs/phase-b0-implementation-manifest.md`;
4. `docs/phase-b0-development-command.md`;
5. `docs/planner-score-development-results.md`;
6. `docs/planner-score-audit-results.md`;
7. `docs/planner-score-alignment-protocol.md`;
8. `docs/latent-planner-controlled-results.md`;
9. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every formal one-step criterion passed.

The controlled multi-step latent planner completed with implementation integrity passed but failed one required criterion. Latent ranking improved all six targets and beat random by 38.10%, but its mean final MSE was 1.996× exact pixel, above the frozen 1.5× maximum.

The Stage A score audit selected MSE-only plus normalized-latent L1. In long-horizon development, forced L1 reduced mean final MSE by about 7.05% versus forced latent MSE and was lower on all three development targets. The zero-margin no-op stopped after only 3.33 strokes on average and failed both scientific eligibility criteria.

## Active task

Phase B0 has a frozen protocol on branch `phase-b/saliency-latent`. Its architecture gate passed, and the complete guarded runner passed 116 local tests plus unauthorized side-effect validation. The separate authorization record now permits exactly one execution of `experiments/21_phase_b_development.py --development`.

## Phase B0 scientific purpose

Test a multi-scale action-conditioned joint-embedding model with rendered spatial action conditioning and a calibrated exact-progress head. B0 remains 64×64 grayscale with the existing renderer so model and objective effects are isolated. Saliency scheduling and color are later, separately frozen phases.

## Hard boundaries

- Do not rerun or tune any completed experiment.
- Do not train or fine-tune against any completed target.
- Do not change or overwrite any completed result.
- Do not tune the Stage A no-op margin or run its reserved confirmatory phase.
- Phase B0 development is authorized for one execution only.
- Run only `caffeinate -dimsu python experiments/21_phase_b_development.py --development` after the authorization-phase tests pass.
- Do not start a second development execution.
- Preserve any `.incomplete` output after an interruption or failure; never delete it to force a retry.
- Formal Phase B0, saliency scheduling B1, and RGB/high-resolution B2 remain unauthorized.
- Do not alter the frozen architecture, objectives, seeds, thresholds, or compute cap.
- Preserve positive and negative outcomes.
- Do not call the approach a canonical JEPA.
