# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/phase-b-saliency-latent-protocol.md`;
3. `docs/phase-b0-implementation-manifest.md`;
4. `configs/phase-b0-aborted-local-attempt-2026-08-23.json`;
5. `docs/phase-b0-colab-preflight-results.md`;
6. `docs/phase-b0-development-command.md`;
7. `docs/phase-b0-colab-preflight.md`;
8. `docs/planner-score-development-results.md`;
9. `docs/planner-score-audit-results.md`;
10. `docs/planner-score-alignment-protocol.md`;
11. `docs/latent-planner-controlled-results.md`;
12. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every formal one-step criterion passed.

The controlled multi-step latent planner completed with implementation integrity passed but failed one required criterion. Latent ranking improved all six targets and beat random by 38.10%, but its mean final MSE was 1.996× exact pixel, above the frozen 1.5× maximum.

The Stage A score audit selected MSE-only plus normalized-latent L1. In long-horizon development, forced L1 reduced mean final MSE by about 7.05% versus forced latent MSE and was lower on all three development targets. The zero-margin no-op stopped after only 3.33 strokes on average and failed both scientific eligibility criteria.

## Active task

The initial Phase B0 local attempt was interrupted before any completed variant and is archived by hash. The original authorization is consumed. The Google Colab CUDA preflight passed on a Tesla T4 with 120 tests, exact resource/state hashes, CPU/CUDA tolerance, finite dummy gradients, and a conservative training-only safety estimate under 0.4 hours. Implement and validate the guarded CUDA recovery runner while recovery remains unauthorized.

## Phase B0 scientific purpose

Test a multi-scale action-conditioned joint-embedding model with rendered spatial action conditioning and a calibrated exact-progress head. B0 remains 64×64 grayscale with the existing renderer so model and objective effects are isolated. Saliency scheduling and color are later, separately frozen phases.

## Hard boundaries

- Do not rerun or tune any completed experiment.
- Do not train or fine-tune against any completed target.
- Do not change or overwrite any completed result.
- Do not tune the Stage A no-op margin or run its reserved confirmatory phase.
- Do not delete, rename, modify, or select against the preserved Phase B0 `.incomplete` directory.
- Phase B0 recovery is unauthorized.
- Do not run `experiments/21_phase_b_development.py --development` locally or in the cloud.
- CUDA recovery implementation may use deterministic random dummy tensors and load frozen resources for hash verification only.
- Do not generate renderer transitions, targets, state banks, candidate sets, scientific checkpoints, or scientific outputs before a separate recovery authorization.
- Do not treat preflight losses or timing as scientific evidence.
- Preserve the frozen architecture, objectives, seeds, thresholds, method order, and six-hour cap.
- Formal Phase B0, saliency scheduling B1, and RGB/high-resolution B2 remain unauthorized.
- Preserve positive and negative outcomes.
- Do not call the approach a canonical JEPA.
