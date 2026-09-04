# Current State

**Last updated:** 2026-09-04  
**Branch:** `quadratic-bezier-extension`  
**Current stage:** fail-closed comparison runner awaiting validation  
**Status:** target hashes frozen; comparative execution unauthorized; no comparative outputs generated or viewed

## Closed evidence

The formal ranking-aware one-step comparison remains a strong positive result at 74.44% retrieval versus 31.44% for MSE-only.

The controlled long-horizon evidence remains closed. Latent ranking improved over random but missed the exact-pixel ratio criterion. The multi-scale prediction-only model achieved 97.61% four-way retrieval, while qualitative 128-candidate exact top-1 was 14% and the trajectory overpainted after step 50.

The exact-pixel RGB baseline and resolution-by-budget ablation are closed. The quality-priority 128x128/420 setting improved mean common-resolution MSE by 17.97% across the fixed five-target set at a 3.56x compute proxy.

## Manuscript state

All six ULAB-aligned chapters have complete v0.1 drafts in Notion. The pre-extension snapshot is dated 2026-09-03. Closed experiment conclusions cannot be rewritten by the new renderer study.

## Active bounded extension

The only active pre-defense experiment compares straight opaque lines with quadratic Bezier curves under exact-pixel RGB selection. It uses six deterministic procedural rights-safe targets, 128x128 planning, 512x512 evaluation, 420 accepted strokes, 64 candidates per pool, and seeds 73/137/211.

The initial scaffold validation passed all 199 tests in 61.78 seconds. The exact target definitions, individual pixel hashes, target order, target-stream mapping, seed order, ordered target-set hash, and decision rule are frozen in `configs/quadratic-bezier-target-freeze-2026-09-04.json`.

Fail-closed runner source commit `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b` now implements the fixed 36-run schedule, complete summaries, common-resolution metrics, candidate-render and runtime accounting, artifact hashes, rights-safe plots, deterministic blinded review materials, and `.incomplete` lifecycle. It checks a separate authorization before creating any output. No authorization file exists.

## Next action

Open `notebooks/quadratic_bezier_comparison_runner_validation.ipynb` from the public branch and run all six cells on a standard Colab CPU runtime. Return `quadratic_bezier_runner_pytest.txt`, `quadratic_bezier_runner_validation.json`, and `quadratic_bezier_unauthorized_probe.txt`. Only after a clean pass may a separate commit authorize one fixed output directory and one completed comparison.
