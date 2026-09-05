# Straight-line versus quadratic-Bezier renderer protocol

**Frozen:** 2026-09-03, before comparative outputs  
**Target manifest frozen:** 2026-09-04, after validation and before runner implementation  
**Branch:** `quadratic-bezier-extension`  
**Base commit:** `d5f1190ab9b62d5adff7fb56c5cc1ffd4d850177`  
**Validated implementation commit:** `7bdcd2e847ca7c5a1faf8a086b26441d8de1a4e1`  
**Current status:** target hashes frozen; comparative execution remains unauthorized

## Purpose

Test whether quadratic Bezier strokes materially improve the attainable quality of the exact-pixel RGB painter relative to straight opaque lines. This is a renderer-capacity experiment. No learned model is trained or loaded in the primary comparison.

## Matched conditions

- condition 1: opaque straight line with start, end, width, and analytically fitted RGB color;
- condition 2: opaque quadratic Bezier with start, control, end, width, and analytically fitted RGB color;
- 128x128 planning and 512x512 common-resolution evaluation;
- 420 accepted strokes split 80/140/200 across global, structure, and detail stages;
- 64 candidates per pool;
- 80/20 error-guided and uniform proposals;
- exact target-pixel MSE selection;
- strict improvement tolerance `1e-9`;
- patience 12;
- seeds 73, 137, and 211;
- identical target order and target-stream mapping.

Both methods receive the same accepted-stroke and candidate-pool budgets. Candidate renders and wall-clock time must also be reported because curve rasterization may cost more.

## Validation evidence

The complete test suite passed with `199 passed in 61.78s`. The no-output validation status was `quadratic_bezier_extension_valid_no_outputs`. Both tiny primitive planners were deterministic and strictly monotonic. The report recorded NumPy 2.1.3 and Pillow 11.3.0, no output side effects, no comparative outputs viewed, no training, no learned model, and no closed-experiment changes.

Exact evidence is preserved in:

- `docs/quadratic-bezier-pytest-2026-09-04.txt`;
- `docs/quadratic-bezier-validation-2026-09-04.json`;
- `configs/quadratic-bezier-target-freeze-2026-09-04.json`.

## Frozen rights-safe target set

The target generator and the following 512x512 RGB pixel hashes are now immutable for this comparison:

- `01_ring_symbol`: `548ee3d03644308c066f11be64234db8a28a67e0a24cb5d21bec6bd6aab4940b`;
- `02_curved_glyph`: `b2e0036f2eb1b4275fb553ae970ab2a93ffb08b229b21b7ebe2dd194e2d0a7da`;
- `03_organic_silhouette`: `838edf87ab05d3b289af18673c87d51e5cd56f77ebef4aa6bb99c7f7685af398`;
- `04_mixed_geometry`: `d660b8247c098d4fc3dac9b51330678371d3efaa73406f86cbfacc6762357f66`;
- `05_layered_landscape`: `4bd0794c85c2be198ee93a2ab0d155ea0758ae0f84bd56fac8428da0e604b45a`;
- `06_dense_scene`: `79d51333fc6d94b2e269cc3a895c0dba2fe12ca2f01d8e9565ef691614c3e3fe`.

Ordered target-set SHA-256: `26bada941bfd8f49f09333d70d397364e82f5ddbb6e1228324f24fb9d2b30bfd`.

## Decision rule

Quadratic Bezier is a `material_improvement` only if all conditions hold:

1. mean final 512 MSE is at least 5% lower across all target-seed pairs;
2. at least four of six per-target means improve;
3. no target mean is more than 5% worse;
4. integrity, determinism, and monotonicity pass;
5. blinded qualitative review finds no systematic regression.

A positive aggregate gain below 5% is `minor_improvement`. No aggregate gain or an integrity failure is `no_material_improvement`. Machine output cannot fill the final qualitative decision.

## Current gate

Target definitions and hashes are frozen, but execution is still unauthorized. The next source change may add a fail-closed comparison runner and validation tests. It must not enable or execute the comparison. A later authorization commit may permit exactly one completed execution only after the runner passes a fresh no-output validation.

## Hard boundaries

- Do not rerun or rewrite a closed experiment.
- Do not use the five private web-sourced RGB targets.
- Do not change the frozen target generator, order, mapping, hashes, seed order, matched settings, or decision thresholds.
- Do not add variable width, alpha, texture, erasing, cubic curves, or mixed primitive sets to the primary comparison.
- Do not train or load a learned model before the exact-pixel decision is complete.
- Do not commit generated binary outputs.
