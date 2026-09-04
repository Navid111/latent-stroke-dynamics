# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/quadratic-bezier-interrupted-run-recovery.md`;
3. `docs/quadratic-bezier-interrupted-recovery-implementation.md`;
4. `configs/quadratic-bezier-interrupted-recovery-plan-2026-09-05.json`;
5. `docs/quadratic-bezier-interrupted-recovery-validation.md`;
6. `docs/quadratic-bezier-interrupted-recovery-validation-results.md`;
7. `docs/quadratic-bezier-interrupted-recovery-authorization.md`;
8. `configs/quadratic-bezier-recovery-authorization-2026-09-05.json`;
9. `docs/quadratic-bezier-interrupted-recovery-execution.md`;
10. `docs/quadratic-bezier-extension-protocol.md`;
11. `configs/quadratic-bezier-target-freeze-2026-09-04.json`;
12. `configs/quadratic-bezier-runner-environment-2026-09-04.json`;
13. `configs/quadratic-bezier-execution-authorization-2026-09-04.json`;
14. `docs/quadratic-bezier-extension-implementation.md`;
15. `configs/quadratic-bezier-extension-2026-09-03.json`;
16. `docs/phase-b-saliency-latent-protocol.md`;
17. `docs/phase-b0-implementation-manifest.md`;
18. `configs/phase-b0-aborted-local-attempt-2026-08-23.json`;
19. `docs/phase-b0-colab-preflight-results.md`;
20. `docs/phase-b0-colab-recovery-protocol.md`;
21. `configs/phase-b0-colab-recovery-2026-08-24.json`;
22. `configs/phase-b0-colab-recovery-authorization-2026-08-24.json`;
23. `docs/phase-b0-colab-recovery-authorization-2026-08-24.md`;
24. `docs/phase-b0-colab-recovery-implementation-manifest.md`;
25. `docs/phase-b0-colab-recovery-local-validation-2026-08-24.md`;
26. `docs/phase-b0-colab-recovery-validation.md`;
27. `docs/phase-b0-colab-recovery-validation-bundle-2026-08-24.md`;
28. `docs/phase-b0-colab-recovery-validation-results.md`;
29. `docs/phase-b0-colab-recovery-execution-handoff.md`;
30. `docs/phase-b0-colab-recovery-command.md`;
31. `docs/phase-b0-development-command.md`;
32. `docs/phase-b0-colab-preflight.md`;
33. `docs/planner-score-development-results.md`;
34. `docs/planner-score-audit-results.md`;
35. `docs/planner-score-alignment-protocol.md`;
36. `docs/latent-planner-controlled-results.md`;
37. relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every held-out one-step criterion passed.

The controlled multi-step latent planner remains a frozen criterion failure with implementation integrity passed. The multi-scale prediction-only model achieved 97.61% four-way retrieval, but its qualitative 128-candidate planner selected the exact best candidate 14% of the time and overpainted after its best frame.

The exact-pixel RGB baseline and resolution-by-budget ablation are complete and immutable. The 128x128/420 condition improved mean common-resolution MSE by 17.97% over 96x96/210 at a 3.56x compute proxy.

## Active task

Run `notebooks/quadratic_bezier_interrupted_recovery_execution.ipynb` once in a fresh CPU Colab runtime. It pins recovery implementation commit `46e0c6396f0425ed84812e8fbeef9ed675ef53e9` and authorization commit `76b6d53bddaaa60880e7c7f1eaffd1392c9ece25`. Run code cells 1–6 in order. Do not interrupt Cell 5. If the browser disconnects, do not press Stop; reconnect later and let the existing runtime continue.

## Hard boundaries

- Do not rerun the old Cell 5 or start a fresh comparison.
- Do not delete, overwrite, manually alter, or inspect comparative content in the preserved `.incomplete` output.
- The authorized recovery may run once only through the pinned recovery execution notebook.
- It may reuse only the 17 byte-verified completed units, must quarantine the partial unit without overwrite, and may execute only the 19 missing units in frozen order.
- If recovery is interrupted or fails, preserve all state and audit again; automatic resume and authorization reuse are prohibited.
- If blinded review is required, do not open metrics, method mappings, numerical plots, aggregate summaries, or logs before completing and recording the review.
- Do not rerun, tune, overwrite, or reinterpret any completed historical experiment.
- Do not modify archived outputs or the preserved incomplete multi-scale recovery directory.
- Do not use the private five-target RGB set in the curve study.
- Do not change frozen targets, hashes, order, target-stream mapping, seeds, matched settings, or decision thresholds.
- Do not add variable width, transparency, texture, erasing, cubic curves, or mixed primitives to the primary comparison.
- Do not train or load a learned model in the primary comparison.
- Do not authorize a learned curved-stroke predictor unless the exact-pixel curve condition first produces a material improvement under the frozen rule.
- Do not commit source images or generated binary outputs.
- Preserve positive and negative outcomes.
- Do not call the approach a canonical JEPA.
