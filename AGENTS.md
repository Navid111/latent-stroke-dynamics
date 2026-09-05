# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Current checkpoint — 2026-09-05

The straight-line versus quadratic-Bezier comparison is complete and closed as `minor_improvement`, not material improvement. Both the original execution and missing-only recovery authorizations are consumed. Never rerun either execution Cell 5. No learned curved-stroke predictor is authorized by this result.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/quadratic-bezier-fixed-comparison-results.md`;
3. `docs/quadratic-bezier-fixed-comparison-result-record-2026-09-05.json`;
4. `docs/quadratic-bezier-extension-protocol.md`;
5. the relevant immutable source and historical validation/recovery records below.

## Historical provenance references — not new execution instructions

- `docs/quadratic-bezier-interrupted-run-recovery.md`;
- `docs/quadratic-bezier-interrupted-recovery-implementation.md`;
- `configs/quadratic-bezier-interrupted-recovery-plan-2026-09-05.json`;
- `docs/quadratic-bezier-interrupted-recovery-validation.md`;
- `docs/quadratic-bezier-interrupted-recovery-validation-results.md`;
- `docs/quadratic-bezier-interrupted-recovery-authorization.md`;
- `configs/quadratic-bezier-recovery-authorization-2026-09-05.json`;
- `docs/quadratic-bezier-interrupted-recovery-execution.md`;
- `configs/quadratic-bezier-target-freeze-2026-09-04.json`;
- `configs/quadratic-bezier-runner-environment-2026-09-04.json`;
- `configs/quadratic-bezier-execution-authorization-2026-09-04.json`;
- `docs/quadratic-bezier-extension-implementation.md`;
- `configs/quadratic-bezier-extension-2026-09-03.json`;
- `docs/phase-b-saliency-latent-protocol.md`;
- `docs/phase-b0-implementation-manifest.md`;
- `configs/phase-b0-aborted-local-attempt-2026-08-23.json`;
- `docs/phase-b0-colab-preflight-results.md`;
- `docs/phase-b0-colab-recovery-protocol.md`;
- `configs/phase-b0-colab-recovery-2026-08-24.json`;
- `configs/phase-b0-colab-recovery-authorization-2026-08-24.json`;
- `docs/phase-b0-colab-recovery-authorization-2026-08-24.md`;
- `docs/phase-b0-colab-recovery-implementation-manifest.md`;
- `docs/phase-b0-colab-recovery-local-validation-2026-08-24.md`;
- `docs/phase-b0-colab-recovery-validation.md`;
- `docs/phase-b0-colab-recovery-validation-bundle-2026-08-24.md`;
- `docs/phase-b0-colab-recovery-validation-results.md`;
- `docs/phase-b0-colab-recovery-execution-handoff.md`;
- `docs/phase-b0-colab-recovery-command.md`;
- `docs/phase-b0-development-command.md`;
- `docs/phase-b0-colab-preflight.md`;
- `docs/planner-score-development-results.md`;
- `docs/planner-score-audit-results.md`;
- `docs/planner-score-alignment-protocol.md`;
- `docs/latent-planner-controlled-results.md`;
- relevant result artifacts and code.

## Frozen evidence

The formal ranking-aware comparison remains immutable: 74.44% retrieval versus 31.44% for MSE-only, +43.00 points, with every held-out one-step criterion passed.

The controlled multi-step latent planner remains a frozen criterion failure with implementation integrity passed. The multi-scale prediction-only model achieved 97.61% four-way retrieval, but its qualitative 128-candidate planner selected the exact best candidate 14% of the time and overpainted after its best frame.

The exact-pixel RGB baseline and resolution-by-budget ablation are complete and immutable. The 128x128/420 condition improved mean common-resolution MSE by 17.97% over 96x96/210 at a 3.56x compute proxy.

The separate six-target curve comparison improved mean final 512 MSE by only 0.7425329829873983%. Four of six target means improved, but the dense scene worsened by 6.518152081149964%. The >=5% aggregate gain and <=5% worst-target worsening requirements failed. Preserve the exact `minor_improvement` category and both failed criteria. Do not rewrite the frozen protocol or original machine output to obtain a different category.

## Active task

Review the text-only result archive and the existing PR #1, then continue manuscript refinement, source-code parameter audit, rights-safe figure layout, citation reconciliation, final assembly, and defense preparation. Notion remains the canonical manuscript; its preserved pre-extension snapshot must not be modified. PR readiness is not permission to merge or to execute an experiment.

No new painting run or test execution was performed for this documentation-only result-archival commit. The latest supplied execution-time test record is 217 passed in 57.21s at the pinned execution handoff; distinguish it from the earlier no-output validation timing.

## Hard boundaries

- Never rerun the old comparison Cell 5 or the recovery Cell 5; both authorizations are consumed.
- Do not start a fresh comparison, recreate the former incomplete directory, automatically resume, or tune a closed result.
- Preserve all 17 reused units, 19 recovered units, the final aggregate, and the quarantined partial unit byte-for-byte.
- Do not repair the raw recovery journal's pre-final in-progress status. Completion is documented separately by the final aggregate and completion handoff.
- No mandatory blinded review was triggered. Numerical exposure preceded descriptive montage inspection; do not claim an independent blinded visual pass.
- The primary statistic is the ratio of overall mean final 512 MSEs, not the mean percentage change. The worsening guard applies to each target's three-seed mean, not its worst seed.
- The progress plot uses 128x128 planning MSE, not the primary final 512x512 replay metric.
- Distinguish notebook-reported verification of 453 artifact hashes from reviewer-side source/metadata/arithmetic checks. No independent rehash of all uploaded/Drive bytes was performed during archival.
- Do not rerun, tune, overwrite, or reinterpret any completed historical experiment.
- Do not modify archived outputs or the preserved incomplete multi-scale recovery directory.
- Do not use the older private five-target RGB set as the six-target curve benchmark.
- Do not change frozen targets, hashes, order, target-stream mapping, seeds, matched settings, or decision thresholds.
- Do not add variable width, transparency, texture, erasing, cubic curves, or mixed primitives to the primary comparison.
- Do not train or load a learned model in this comparison. The observed non-material result does not activate a learned curved-stroke extension.
- Publish only the approved compact text evidence: no source images, generated binaries, private manuscript text, private attachment URLs, or secrets.
- Preserve positive and negative outcomes and their scope. Do not call the approach a canonical JEPA.
