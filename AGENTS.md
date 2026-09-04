# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/quadratic-bezier-extension-protocol.md`;
3. `configs/quadratic-bezier-target-freeze-2026-09-04.json`;
4. `docs/quadratic-bezier-extension-implementation.md`;
5. `configs/quadratic-bezier-extension-2026-09-03.json`;
6. `docs/phase-b-saliency-latent-protocol.md`;
7. `docs/phase-b0-implementation-manifest.md`;
8. `configs/phase-b0-aborted-local-attempt-2026-08-23.json`;
9. `docs/phase-b0-colab-preflight-results.md`;
10. `docs/phase-b0-colab-recovery-protocol.md`;
11. `configs/phase-b0-colab-recovery-2026-08-24.json`;
12. `configs/phase-b0-colab-recovery-authorization-2026-08-24.json`;
13. `docs/phase-b0-colab-recovery-authorization-2026-08-24.md`;
14. `docs/phase-b0-colab-recovery-implementation-manifest.md`;
15. `docs/phase-b0-colab-recovery-local-validation-2026-08-24.md`;
16. `docs/phase-b0-colab-recovery-validation.md`;
17. `docs/phase-b0-colab-recovery-validation-bundle-2026-08-24.md`;
18. `docs/phase-b0-colab-recovery-validation-results.md`;
19. `docs/phase-b0-colab-recovery-execution-handoff.md`;
20. `docs/phase-b0-colab-recovery-command.md`;
21. `docs/phase-b0-development-command.md`;
22. `docs/phase-b0-colab-preflight.md`;
23. `docs/planner-score-development-results.md`;
24. `docs/planner-score-audit-results.md`;
25. `docs/planner-score-alignment-protocol.md`;
26. `docs/latent-planner-controlled-results.md`;
27. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every held-out one-step criterion passed.

The controlled multi-step latent planner remains a frozen criterion failure with implementation integrity passed. The multi-scale prediction-only model achieved 97.61% four-way retrieval, but its qualitative 128-candidate planner selected the exact best candidate 14% of the time and overpainted after its best frame.

The exact-pixel RGB baseline and resolution-by-budget ablation are complete and immutable. The 128x128/420 condition improved mean common-resolution MSE by 17.97% over 96x96/210 at a 3.56x compute proxy.

## Active task

Implement and validate exactly one fail-closed runner for the bounded straight-line versus quadratic-Bezier comparison. The six procedural targets, target order, target-stream mapping, seeds, matched settings, hashes, and decision rule are frozen in `configs/quadratic-bezier-target-freeze-2026-09-04.json`. Comparative execution remains unauthorized and no outputs may be viewed yet.

## Hard boundaries

- Do not rerun, tune, overwrite, or reinterpret any completed experiment.
- Do not modify archived outputs or the preserved incomplete recovery directory.
- Do not use the private five-target RGB set in the curve study.
- Do not view comparative straight-versus-curve outputs before the runner implementation, complete tests, environment manifest, authorization config, and exact source commit are frozen.
- Do not change the frozen targets, hashes, order, target-stream mapping, seeds, matched settings, or decision thresholds.
- Do not add variable width, transparency, texture, erasing, cubic curves, or mixed primitives to the primary comparison.
- Do not train or load a learned model in the primary comparison.
- Do not authorize a learned curved-stroke predictor unless the exact-pixel curve condition first produces a material improvement under the frozen rule.
- Do not commit source images or generated binary outputs.
- Preserve positive and negative outcomes.
- Do not call the approach a canonical JEPA.
