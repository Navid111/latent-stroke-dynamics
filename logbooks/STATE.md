# Current State

**Last updated:** 2026-09-03  
**Branch:** `quadratic-bezier-extension`  
**Current stage:** validation-only quadratic-Bezier renderer implementation  
**Status:** six-chapter v0.1 preserved; comparative execution unauthorized

## Closed evidence

The formal ranking-aware one-step comparison remains a strong positive result at 74.44% retrieval versus 31.44% for MSE-only.

The controlled long-horizon evidence remains closed. Latent ranking improved over random but missed the exact-pixel ratio criterion. The multi-scale prediction-only model achieved 97.61% four-way retrieval, while qualitative 128-candidate exact top-1 was 14% and the trajectory overpainted after step 50.

The exact-pixel RGB baseline and resolution-by-budget ablation are closed. The quality-priority 128x128/420 setting improved mean common-resolution MSE by 17.97% across the fixed five-target set at a 3.56x compute proxy.

## Manuscript state

All six ULAB-aligned chapters have complete v0.1 drafts in Notion. The pre-extension snapshot is dated 2026-09-03. Closed experiment conclusions cannot be rewritten by the new renderer study.

## Active bounded extension

The only active pre-defense experiment compares straight opaque lines with quadratic Bezier curves under exact-pixel RGB selection. It uses six deterministic procedural rights-safe targets, 128x128 planning, 512x512 evaluation, 420 accepted strokes, 64 candidates per pool, and seeds 73/137/211.

The initial implementation commit is validation-only. Target hashes, execution authorization, and comparative outputs remain unfrozen or prohibited. No learned model is permitted.

## Next action

Run the complete local test suite and `python validate_quadratic_bezier_extension.py --validate-only`. Review the six generated target hashes, deterministic and monotonic smoke results, and no-side-effect boundary. Only after a passing report may a separate commit freeze target hashes and add or authorize the one fixed comparative runner.
