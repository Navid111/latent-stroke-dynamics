# Current State

**Last updated:** 2026-09-04  
**Branch:** `quadratic-bezier-extension`  
**Current stage:** target manifest frozen; comparison runner not yet authorized  
**Status:** six-chapter v0.1 preserved; no comparative outputs generated or viewed

## Closed evidence

The formal ranking-aware one-step comparison remains a strong positive result at 74.44% retrieval versus 31.44% for MSE-only.

The controlled long-horizon evidence remains closed. Latent ranking improved over random but missed the exact-pixel ratio criterion. The multi-scale prediction-only model achieved 97.61% four-way retrieval, while qualitative 128-candidate exact top-1 was 14% and the trajectory overpainted after step 50.

The exact-pixel RGB baseline and resolution-by-budget ablation are closed. The quality-priority 128x128/420 setting improved mean common-resolution MSE by 17.97% across the fixed five-target set at a 3.56x compute proxy.

## Manuscript state

All six ULAB-aligned chapters have complete v0.1 drafts in Notion. The pre-extension snapshot is dated 2026-09-03. Closed experiment conclusions cannot be rewritten by the new renderer study.

## Active bounded extension

The only active pre-defense experiment compares straight opaque lines with quadratic Bezier curves under exact-pixel RGB selection. It uses six deterministic procedural rights-safe targets, 128x128 planning, 512x512 evaluation, 420 accepted strokes, 64 candidates per pool, and seeds 73/137/211.

Validation of implementation commit `7bdcd2e847ca7c5a1faf8a086b26441d8de1a4e1` passed all 199 tests in 61.78 seconds. Both primitive smoke planners were deterministic and monotonic. No output side effects, training, learned model, comparative-output viewing, or changes to closed experiments were recorded.

The exact target definitions, individual pixel hashes, target order, target-stream mapping, seed order, ordered target-set hash, and decision rule are now frozen in `configs/quadratic-bezier-target-freeze-2026-09-04.json`. The ordered target-set hash is `26bada941bfd8f49f09333d70d397364e82f5ddbb6e1228324f24fb9d2b30bfd`.

## Next action

Implement a fail-closed comparison runner while keeping execution unauthorized. It must report all target-seed-condition summaries, common 512x512 RGB MSE, per-target means and ratios, candidate-render counts, wall-clock time, artifact hashes, rights-safe aggregate plots and montages, blinded labels, and an `.incomplete` lifecycle. Run a fresh no-output validation before creating a separate one-time execution authorization.
