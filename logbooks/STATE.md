# Current State

**Last updated:** 2026-09-04  
**Branch:** `quadratic-bezier-extension`  
**Current stage:** one fixed comparison authorized and awaiting execution  
**Status:** target hashes and runner environment frozen; no comparative outputs generated or viewed

## Closed evidence

The formal ranking-aware one-step comparison remains a strong positive result at 74.44% retrieval versus 31.44% for MSE-only.

The controlled long-horizon evidence remains closed. Latent ranking improved over random but missed the exact-pixel ratio criterion. The multi-scale prediction-only model achieved 97.61% four-way retrieval, while qualitative 128-candidate exact top-1 was 14% and the trajectory overpainted after step 50.

The exact-pixel RGB baseline and resolution-by-budget ablation are closed. The quality-priority 128x128/420 setting improved mean common-resolution MSE by 17.97% across the fixed five-target set at a 3.56x compute proxy.

## Manuscript state

All six ULAB-aligned chapters have complete v0.1 drafts in Notion. The pre-extension snapshot is dated 2026-09-03. Closed experiment conclusions cannot be rewritten by the new renderer study.

## Active bounded extension

The only active pre-defense experiment compares straight opaque lines with quadratic Bezier curves under exact-pixel RGB selection. It uses six deterministic procedural rights-safe targets, 128x128 planning, 512x512 evaluation, 420 accepted strokes, 64 candidates per pool, and seeds 73/137/211.

The scaffold passed 199 tests before the six target hashes were frozen. Fail-closed runner source commit `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b` then passed all 210 tests in 60.84 seconds in Google Colab. Its validation report captured Python 3.13.15, Linux, NumPy 2.1.3, Pillow 11.3.0, and Matplotlib 3.10.0. Both primitive smoke planners were deterministic and monotonic. The unauthorized execution probe correctly raised `PermissionError` and created no output directory.

Authorization commit `cc857407ed431c5583fd9e1c02a0ba619a8c187a` permits one completed execution at `/content/drive/MyDrive/latent-stroke-dynamics-rgb/quadratic-bezier-fixed-comparison-v1`. No learned model is permitted. An incomplete attempt must be preserved and reviewed before any recovery decision.

## Next action

Open `notebooks/quadratic_bezier_fixed_comparison_execution.ipynb` and run code cells 1–6 in order. Do not interrupt. If Cell 6 says blinded review is required, return only the blind handoff JSON, blinded montage, and blank review sheet; do not run Cell 7. If Cell 6 says blinded review is not required, return the downloaded numerical handoff. Do not rerun the comparison.
