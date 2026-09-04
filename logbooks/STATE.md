# Current State

**Last updated:** 2026-09-05  
**Branch:** `quadratic-bezier-extension`  
**Current stage:** one guarded missing-only recovery authorized and ready for execution  
**Status:** no completed comparative result; do not use the old execution notebook

## Closed evidence

The formal ranking-aware one-step comparison remains a strong positive result at 74.44% retrieval versus 31.44% for MSE-only.

The controlled long-horizon evidence remains closed. Latent ranking improved over random but missed the exact-pixel ratio criterion. The multi-scale prediction-only model achieved 97.61% four-way retrieval, while qualitative 128-candidate exact top-1 was 14% and the trajectory overpainted after step 50.

The exact-pixel RGB baseline and resolution-by-budget ablation are closed. The quality-priority 128x128/420 setting improved mean common-resolution MSE by 17.97% across the fixed five-target set at a 3.56x compute proxy.

## Manuscript state

All six ULAB-aligned chapters have complete v0.1 drafts in Notion. The pre-extension snapshot is dated 2026-09-03. Closed experiment conclusions cannot be rewritten by the new renderer study.

## Active bounded extension

The fixed comparison uses six deterministic procedural rights-safe targets, 128x128 planning, 512x512 evaluation, 420 accepted strokes, 64 candidates per pool, and seeds 73/137/211. Target hashes, source, tests, environment, and decision rules remain frozen.

Fail-closed runner source commit `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b` passed all 210 tests and the unauthorized-output probe. Original authorization commit `cc857407ed431c5583fd9e1c02a0ba619a8c187a` permits one completed comparison.

## Stable interrupted state

Two identical read-only diagnostics were captured, with the second taken only after the original Colab runtime was disconnected and deleted. The stable state has no completed output or aggregate, 17 valid completed units, one partial unit (`03_organic_silhouette/seed_211/quadratic_bezier`), and 18 units that never started. No images or metrics were opened.

## Recovery validation and authorization

Recovery implementation commit `46e0c6396f0425ed84812e8fbeef9ed675ef53e9` passed 217 tests and the explicit fail-closed probe without mounting Drive or accessing the interrupted output. Evidence is archived in commit `5cc2e6c98bb58b6ad917b593b97dbd359033fe75`.

Authorization commit `76b6d53bddaaa60880e7c7f1eaffd1392c9ece25` permits one missing-only recovery. It must byte-preserve the 17 completed units, quarantine the partial unit, execute 19 missing units in frozen order, rebuild one aggregate after all 36 units verify, and apply the blind gate.

## Next action

Disconnect and delete the validation runtime after its three files are safely downloaded. Open `notebooks/quadratic_bezier_interrupted_recovery_execution.ipynb` in a fresh CPU runtime and run code cells 1–6 in order. Do not interrupt Cell 5. If the browser disconnects or shows reconnecting, do not press Stop; reconnect later and let the runtime continue. If Cell 6 requires blinded review, return only its three blind-review files and do not run Cell 7.
