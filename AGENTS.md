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
8. `configs/phase-b0-colab-recovery-authorization-2026-08-24.json`;
9. `docs/phase-b0-colab-recovery-authorization-2026-08-24.md`;
10. `docs/phase-b0-colab-recovery-implementation-manifest.md`;
11. `docs/phase-b0-colab-recovery-local-validation-2026-08-24.md`;
12. `docs/phase-b0-colab-recovery-validation.md`;
13. `docs/phase-b0-colab-recovery-validation-bundle-2026-08-24.md`;
14. `docs/phase-b0-colab-recovery-validation-results.md`;
15. `docs/phase-b0-colab-recovery-execution-handoff.md`;
16. `docs/phase-b0-colab-recovery-command.md`;
17. `docs/phase-b0-development-command.md`;
18. `docs/phase-b0-colab-preflight.md`;
19. `docs/planner-score-development-results.md`;
20. `docs/planner-score-audit-results.md`;
21. `docs/planner-score-alignment-protocol.md`;
22. `docs/latent-planner-controlled-results.md`;
23. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every formal one-step criterion passed.

The controlled multi-step latent planner completed with implementation integrity passed but failed one required criterion. Latent ranking improved all six targets and beat random by 38.10%, but its mean final MSE was 1.996× exact pixel, above the frozen 1.5× maximum.

The Stage A score audit selected MSE-only plus normalized-latent L1. In long-horizon development, forced L1 reduced mean final MSE by about 7.05% versus forced latent MSE and was lower on all three development targets. The zero-margin no-op stopped after only 3.33 strokes on average and failed both scientific eligibility criteria.

## Active task

Exactly one Phase B0 Colab recovery execution is authorized and unconsumed. Pull the authorization commit, run the 145-test suite, build the authorized execution bundle exactly once, and use only `notebooks/phase_b0_colab_recovery_execution.ipynb` on a fresh Tesla T4. Do not run the recovery locally. Formal B0, B1, and B2 remain unauthorized.

## Hard boundaries

- Do not rerun or tune any completed experiment.
- Do not train or fine-tune against any completed target.
- Do not change or overwrite any completed result.
- Do not tune the Stage A no-op margin or run its reserved confirmatory phase.
- Do not delete, rename, modify, or select against the preserved local Phase B0 `.incomplete` directory.
- The recovery authorization permits exactly one cloud execution and is not reusable.
- Do not run `experiments/21_phase_b_development.py --development`.
- Do not run `experiments/23_phase_b_colab_recovery.py` locally.
- Build the authorized execution bundle only after the 145-test suite passes.
- In Colab, run the readiness check before changing the explicit execution switch to true.
- If interrupted, preserve the Google Drive `.incomplete` directory and console log; do not resume or restart without an audit.
- Do not tune or rerun against the recovery result.
- Formal Phase B0, saliency scheduling B1, and RGB/high-resolution B2 remain unauthorized.
- Preserve positive and negative outcomes.
- Do not call the approach a canonical JEPA.
