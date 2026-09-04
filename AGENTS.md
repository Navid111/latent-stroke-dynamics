# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/quadratic-bezier-interrupted-run-recovery.md`;
3. `docs/quadratic-bezier-extension-protocol.md`;
4. `configs/quadratic-bezier-target-freeze-2026-09-04.json`;
5. `configs/quadratic-bezier-runner-environment-2026-09-04.json`;
6. `configs/quadratic-bezier-execution-authorization-2026-09-04.json`;
7. `docs/quadratic-bezier-extension-implementation.md`;
8. `configs/quadratic-bezier-extension-2026-09-03.json`;
9. `docs/phase-b-saliency-latent-protocol.md`;
10. `docs/phase-b0-implementation-manifest.md`;
11. `configs/phase-b0-aborted-local-attempt-2026-08-23.json`;
12. `docs/phase-b0-colab-preflight-results.md`;
13. `docs/phase-b0-colab-recovery-protocol.md`;
14. `configs/phase-b0-colab-recovery-2026-08-24.json`;
15. `configs/phase-b0-colab-recovery-authorization-2026-08-24.json`;
16. `docs/phase-b0-colab-recovery-authorization-2026-08-24.md`;
17. `docs/phase-b0-colab-recovery-implementation-manifest.md`;
18. `docs/phase-b0-colab-recovery-local-validation-2026-08-24.md`;
19. `docs/phase-b0-colab-recovery-validation.md`;
20. `docs/phase-b0-colab-recovery-validation-bundle-2026-08-24.md`;
21. `docs/phase-b0-colab-recovery-validation-results.md`;
22. `docs/phase-b0-colab-recovery-execution-handoff.md`;
23. `docs/phase-b0-colab-recovery-command.md`;
24. `docs/phase-b0-development-command.md`;
25. `docs/phase-b0-colab-preflight.md`;
26. `docs/planner-score-development-results.md`;
27. `docs/planner-score-audit-results.md`;
28. `docs/planner-score-alignment-protocol.md`;
29. `docs/latent-planner-controlled-results.md`;
30. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every held-out one-step criterion passed.

The controlled multi-step latent planner remains a frozen criterion failure with implementation integrity passed. The multi-scale prediction-only model achieved 97.61% four-way retrieval, but its qualitative 128-candidate planner selected the exact best candidate 14% of the time and overpainted after its best frame.

The exact-pixel RGB baseline and resolution-by-budget ablation are complete and immutable. The 128x128/420 condition improved mean common-resolution MSE by 17.97% over 96x96/210 at a 3.56x compute proxy.

## Active task

Inspect the preserved interrupted attempt with `notebooks/quadratic_bezier_incomplete_run_inspection.ipynb`. The first authorized comparison attempt was manually interrupted after approximately 12 minutes following an internet outage. The notebook handler terminated the child process and should have preserved `quadratic-bezier-fixed-comparison-v1.incomplete`. This is not a completed execution and not a scientific result. No rerun or recovery is authorized until the read-only diagnostic is reviewed.

## Hard boundaries

- Do not rerun Cell 5 or start a new comparison.
- Do not delete, rename, overwrite, or manually alter `quadratic-bezier-fixed-comparison-v1.incomplete`.
- Do not open partial images, numerical summaries, plots, mappings, or infer outcomes from partial completion order.
- Do not design or execute recovery before reviewing `quadratic_bezier_incomplete_diagnostic.json`.
- Do not rerun, tune, overwrite, or reinterpret any completed historical experiment.
- Do not modify archived outputs or the preserved incomplete multi-scale recovery directory.
- Do not use the private five-target RGB set in the curve study.
- Do not change the frozen targets, hashes, order, target-stream mapping, seeds, matched settings, or decision thresholds.
- Any recovery must preserve and verify completed units, quarantine rather than overwrite a partial unit, execute only missing units, and still yield one completed aggregate result.
- Do not add variable width, transparency, texture, erasing, cubic curves, or mixed primitives to the primary comparison.
- Do not train or load a learned model in the primary comparison.
- Do not authorize a learned curved-stroke predictor unless the exact-pixel curve condition first produces a material improvement under the frozen rule.
- Do not commit source images or generated binary outputs.
- Preserve positive and negative outcomes.
- Do not call the approach a canonical JEPA.
