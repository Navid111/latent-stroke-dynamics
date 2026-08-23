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
6. `docs/phase-b0-colab-recovery-protocol.md`;
7. `configs/phase-b0-colab-recovery-2026-08-24.json`;
8. `docs/phase-b0-colab-recovery-implementation-manifest.md`;
9. `docs/phase-b0-colab-recovery-local-validation-2026-08-24.md`;
10. `docs/phase-b0-colab-recovery-validation.md`;
11. `docs/phase-b0-colab-recovery-command.md`;
12. `docs/phase-b0-development-command.md`;
13. `docs/phase-b0-colab-preflight.md`;
14. `docs/planner-score-development-results.md`;
15. `docs/planner-score-audit-results.md`;
16. `docs/planner-score-alignment-protocol.md`;
17. `docs/latent-planner-controlled-results.md`;
18. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every formal one-step criterion passed.

The controlled multi-step latent planner completed with implementation integrity passed but failed one required criterion. Latent ranking improved all six targets and beat random by 38.10%, but its mean final MSE was 1.996× exact pixel, above the frozen 1.5× maximum.

The Stage A score audit selected MSE-only plus normalized-latent L1. In long-horizon development, forced L1 reduced mean final MSE by about 7.05% versus forced latent MSE and was lower on all three development targets. The zero-margin no-op stopped after only 3.33 strokes on average and failed both scientific eligibility criteria.

## Active task

The complete guarded persistent recovery runner passed 132 local tests and validation-only lifecycle checks. The new exact six-resource recovery-validation bundle builder and fail-closed Colab notebook are implemented. Run the updated 138-test local suite, rebuild the bundle from the exact clean head, and run dummy-only CUDA validation on a Tesla T4. Recovery remains unauthorized.

## Hard boundaries

- Do not rerun or tune any completed experiment.
- Do not train or fine-tune against any completed target.
- Do not change or overwrite any completed result.
- Do not tune the Stage A no-op margin or run its reserved confirmatory phase.
- Do not delete, rename, modify, or select against the preserved Phase B0 `.incomplete` directory.
- Phase B0 recovery is unauthorized.
- Do not run `experiments/21_phase_b_development.py --development` locally or in the cloud.
- Do not run `experiments/23_phase_b_colab_recovery.py` in execution mode before a separate recovery-authorization commit.
- Recovery validation may use deterministic random dummy tensors, temporary dummy checkpoints, and frozen resources for hash checks only.
- Do not generate renderer transitions, targets, state banks, candidate sets, recovery outputs, scientific checkpoints, or scientific results before a separate recovery authorization.
- Do not treat preflight or validation losses and timing as scientific evidence.
- Preserve the frozen architecture, objectives, seeds, thresholds, method order, and six-hour cap.
- Formal Phase B0, saliency scheduling B1, and RGB/high-resolution B2 remain unauthorized.
- Preserve positive and negative outcomes.
- Do not call the approach a canonical JEPA.
