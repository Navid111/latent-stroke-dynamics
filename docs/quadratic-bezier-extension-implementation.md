# Quadratic-Bezier extension implementation gate

The validated scaffold provides deterministic quadratic-Bezier rasterization, curve masks, analytically fitted RGB colors, unique changing proposals, matched exact-pixel planning, action serialization, common-resolution replay, and six original procedural targets.

## Completed scaffold validation

Validation used exact implementation commit `7bdcd2e847ca7c5a1faf8a086b26441d8de1a4e1`, reached `199 passed in 61.78s`, and returned `quadratic_bezier_extension_valid_no_outputs`. Straight and quadratic smoke planners were deterministic and monotonic. The report confirmed zero output side effects, no comparative outputs viewed, no training, no learned model, and no changes to closed experiments.

The exact returned files are preserved as:

- `docs/quadratic-bezier-pytest-2026-09-04.txt`;
- `docs/quadratic-bezier-validation-2026-09-04.json`.

## Frozen target manifest

`configs/quadratic-bezier-target-freeze-2026-09-04.json` freezes the six target definitions, target order, zero-based target-stream mapping, seed order, individual 512x512 RGB pixel hashes, ordered target-set hash, validation evidence, and decision thresholds. The ordered target-set hash is `26bada941bfd8f49f09333d70d397364e82f5ddbb6e1228324f24fb9d2b30bfd`.

## Fail-closed comparison runner

Exact runner source commit `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b` adds:

- target-freeze and generator-hash validation;
- a separate one-time authorization boundary checked before output creation;
- all 36 fixed target-seed-condition runs;
- per-run actions, progress, summaries, common-resolution MSE, candidate-render counts, runtimes, and artifact hashes;
- aggregate means, per-target means and ratios, quantitative classification, progress and metric plots;
- a rights-safe, deterministically randomized blinded montage and review sheet;
- an `.incomplete` lifecycle and preserved failure report;
- no learned-model path.

Comparative execution remains unauthorized. The runner must first pass the tokenless notebook `notebooks/quadratic_bezier_comparison_runner_validation.ipynb`. That notebook pins commit `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b`, runs the complete test suite, validates the runner without outputs, and proves that an unauthorized execution attempt creates neither a completed nor an incomplete output directory.

## Current boundary

Do not add an authorization file and do not execute the comparison until the complete runner validation log, validation JSON, and unauthorized-probe log have been reviewed. After a clean pass, create one separate authorization commit that fixes the exact runner commit and output-directory name.
