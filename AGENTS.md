# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/quadratic-bezier-extension-protocol.md`;
3. `configs/quadratic-bezier-extension-2026-09-03.json`;
4. `docs/phase-b-saliency-latent-protocol.md`;
5. `docs/phase-b0-implementation-manifest.md`;
6. `configs/phase-b0-aborted-local-attempt-2026-08-23.json`;
7. `docs/phase-b0-colab-preflight-results.md`;
8. `docs/phase-b0-colab-recovery-protocol.md`;
9. `configs/phase-b0-colab-recovery-2026-08-24.json`;
10. `configs/phase-b0-colab-recovery-authorization-2026-08-24.json`;
11. `docs/phase-b0-colab-recovery-authorization-2026-08-24.md`;
12. `docs/phase-b0-colab-recovery-implementation-manifest.md`;
13. `docs/phase-b0-colab-recovery-local-validation-2026-08-24.md`;
14. `docs/phase-b0-colab-recovery-validation.md`;
15. `docs/phase-b0-colab-recovery-validation-bundle-2026-08-24.md`;
16. `docs/phase-b0-colab-recovery-validation-results.md`;
17. `docs/phase-b0-colab-recovery-execution-handoff.md`;
18. `docs/phase-b0-colab-recovery-command.md`;
19. `docs/phase-b0-development-command.md`;
20. `docs/phase-b0-colab-preflight.md`;
21. `docs/planner-score-development-results.md`;
22. `docs/planner-score-audit-results.md`;
23. `docs/planner-score-alignment-protocol.md`;
24. `docs/latent-planner-controlled-results.md`;
25. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every held-out one-step criterion passed.

The controlled multi-step latent planner remains a frozen criterion failure with implementation integrity passed. The multi-scale prediction-only model achieved 97.61% four-way retrieval, but its qualitative 128-candidate planner selected the exact best candidate 14% of the time and overpainted after its best frame.

The exact-pixel RGB baseline and resolution-by-budget ablation are complete and immutable. The 128x128/420 condition improved mean common-resolution MSE by 17.97% over 96x96/210 at a 3.56x compute proxy.

## Active task

Implement and validate exactly one bounded pre-defense renderer study: straight opaque lines versus opaque quadratic Bezier curves under matched exact-pixel selection. The six-chapter thesis v0.1 is already preserved. The current branch is validation-only: target hashes are not yet frozen and comparative execution is unauthorized.

## Hard boundaries

- Do not rerun, tune, overwrite, or reinterpret any completed experiment.
- Do not modify archived outputs or the preserved incomplete recovery directory.
- Do not use the private five-target RGB set in the curve study.
- Do not view comparative straight-versus-curve outputs before target definitions, target hashes, implementation tests, protocol config, and the source commit are frozen.
- Do not change targets or decision thresholds after comparative outputs are visible.
- Do not add variable width, transparency, texture, erasing, cubic curves, or mixed primitives to the primary comparison.
- Do not train or load a learned model in the primary comparison.
- Do not authorize a learned curved-stroke predictor unless the exact-pixel curve condition first produces a material improvement under the frozen rule.
- Do not commit source images or generated binary outputs.
- Preserve positive and negative outcomes.
- Do not call the approach a canonical JEPA.
