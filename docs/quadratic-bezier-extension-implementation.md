# Quadratic-Bezier extension implementation gate

The validated scaffold provides deterministic quadratic-Bezier rasterization, curve masks, analytically fitted RGB colors, unique changing proposals, matched exact-pixel planning, action serialization, common-resolution replay, and six original procedural targets.

## Completed validation

Validation used exact implementation commit `7bdcd2e847ca7c5a1faf8a086b26441d8de1a4e1`, reached `199 passed in 61.78s`, and returned `quadratic_bezier_extension_valid_no_outputs`. Straight and quadratic smoke planners were deterministic and monotonic. The report confirmed zero output side effects, no comparative outputs viewed, no training, no learned model, and no changes to closed experiments.

The exact returned files are preserved as:

- `docs/quadratic-bezier-pytest-2026-09-04.txt`;
- `docs/quadratic-bezier-validation-2026-09-04.json`.

## Frozen target manifest

`configs/quadratic-bezier-target-freeze-2026-09-04.json` freezes the six target definitions, target order, zero-based target-stream mapping, seed order, individual 512x512 RGB pixel hashes, ordered target-set hash, validation evidence, and decision thresholds. The ordered target-set hash is `26bada941bfd8f49f09333d70d397364e82f5ddbb6e1228324f24fb9d2b30bfd`.

## Current boundary

Comparative execution remains unauthorized. No comparison command or output has been generated or viewed. The next commit may add a fail-closed runner, summaries, aggregate plots, blinded labels, artifact hashing, and validation tests, but it must preserve `execution_authorized: false`. A separate authorization commit is required before the single fixed execution.
