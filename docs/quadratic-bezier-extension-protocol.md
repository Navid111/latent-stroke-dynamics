# Straight-line versus quadratic-Bezier renderer protocol

**Frozen:** 2026-09-03, before comparative implementation outputs  
**Branch:** `quadratic-bezier-extension`  
**Base commit:** `d5f1190ab9b62d5adff7fb56c5cc1ffd4d850177`  
**Current status:** validation-only implementation; target hashes and execution remain unfrozen and unauthorized

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
- identical target order and seed mapping.

Both methods receive the same accepted-stroke and candidate-evaluation budgets. Wall-clock time and candidate renders must also be reported because curve rasterization may cost more.

## Rights-safe target set

Six deterministic procedural targets are defined in source: ring symbol, curved glyph, organic silhouette, mixed geometry, layered landscape, and dense scene. Their pixel hashes are not frozen in the initial implementation commit. The validation report must produce the six hashes and ordered target-set hash. A later commit must record those exact values before any comparative output is generated.

## Decision rule

Quadratic Bezier is a `material_improvement` only if all conditions hold:

1. mean final 512 MSE is at least 5% lower across all target-seed pairs;
2. at least four of six per-target means improve;
3. no target mean is more than 5% worse;
4. integrity, determinism, and monotonicity pass;
5. blinded qualitative review finds no systematic regression.

A positive gain below 5% is `minor_improvement`. No aggregate gain or an integrity failure is `no_material_improvement`. Machine output cannot fill the final qualitative decision.

## Current gate

The current commit may only validate config, deterministic procedural targets, straight and curve renderers, unique changing proposals, serialization, and tiny in-memory planners. It must create no output directory and expose no execution command.

Run:

```bash
python -m pytest
python validate_quadratic_bezier_extension.py --validate-only
```

Return the full test result and validation JSON. Do not run a comparison yet.

## Hard boundaries

- Do not rerun or rewrite a closed experiment.
- Do not use the five private web-sourced RGB targets.
- Do not change target definitions after their hashes are frozen.
- Do not add variable width, alpha, texture, erasing, cubic curves, or mixed primitive sets to the primary comparison.
- Do not train a learned model before the exact-pixel decision is complete.
- Do not commit generated binary outputs.
