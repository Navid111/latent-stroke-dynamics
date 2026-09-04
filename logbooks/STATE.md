# Current State

**Last updated:** 2026-09-05  
**Branch:** `quadratic-bezier-extension`  
**Current stage:** stable interrupted state confirmed; guarded recovery implementation awaiting no-output validation  
**Status:** no completed comparative result; recovery execution unauthorized

## Closed evidence

The formal ranking-aware one-step comparison remains a strong positive result at 74.44% retrieval versus 31.44% for MSE-only.

The controlled long-horizon evidence remains closed. Latent ranking improved over random but missed the exact-pixel ratio criterion. The multi-scale prediction-only model achieved 97.61% four-way retrieval, while qualitative 128-candidate exact top-1 was 14% and the trajectory overpainted after step 50.

The exact-pixel RGB baseline and resolution-by-budget ablation are closed. The quality-priority 128x128/420 setting improved mean common-resolution MSE by 17.97% across the fixed five-target set at a 3.56x compute proxy.

## Manuscript state

All six ULAB-aligned chapters have complete v0.1 drafts in Notion. The pre-extension snapshot is dated 2026-09-03. Closed experiment conclusions cannot be rewritten by the new renderer study.

## Active bounded extension

The fixed comparison uses six deterministic procedural rights-safe targets, 128x128 planning, 512x512 evaluation, 420 accepted strokes, 64 candidates per pool, and seeds 73/137/211. Target hashes, source, tests, environment, and decision rules remain frozen.

Fail-closed runner source commit `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b` passed all 210 tests and the unauthorized-output probe. Authorization commit `cc857407ed431c5583fd9e1c02a0ba619a8c187a` permits one completed comparison.

## Stable interrupted state

Two identical read-only diagnostics were captured, with the second taken only after the original Colab runtime was disconnected and deleted. The stable state has no completed output or aggregate, 17 valid completed units, one partial unit (`03_organic_silhouette/seed_211/quadratic_bezier`), and 18 units that never started. No images or metrics were opened.

The partial unit must be hashed and quarantined without overwrite. The 17 completed units must remain byte-for-byte unchanged. Exactly 19 missing units may eventually run in the original frozen order, after a separate no-output recovery validation and a later one-time authorization.

## Next action

Run the pinned recovery-validation notebook when it is added. It must not mount Drive or access the interrupted output. Archive its JSON report and pytest log. Do not run recovery, reopen the old execution notebook, inspect generated outputs, or modify the preserved `.incomplete` directory.
