# Quadratic-Bezier extension implementation gate

The validated scaffold provides deterministic quadratic-Bezier rasterization, curve masks, analytically fitted RGB colors, unique changing proposals, matched exact-pixel planning, action serialization, common-resolution replay, and six original procedural targets.

## Completed scaffold validation

Validation used exact implementation commit `7bdcd2e847ca7c5a1faf8a086b26441d8de1a4e1`, reached `199 passed in 61.78s`, and returned `quadratic_bezier_extension_valid_no_outputs`. Straight and quadratic smoke planners were deterministic and monotonic. The report confirmed zero output side effects, no comparative outputs viewed, no training, no learned model, and no changes to closed experiments.

## Frozen target manifest

`configs/quadratic-bezier-target-freeze-2026-09-04.json` freezes the six target definitions, target order, zero-based target-stream mapping, seed order, individual 512x512 RGB pixel hashes, ordered target-set hash, validation evidence, and decision thresholds. The ordered target-set hash is `26bada941bfd8f49f09333d70d397364e82f5ddbb6e1228324f24fb9d2b30bfd`.

## Validated fail-closed runner

Exact runner source commit `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b` implements:

- target-freeze and generator-hash validation;
- a separate authorization boundary checked before output creation;
- all 36 fixed target-seed-condition runs;
- per-run actions, progress, common-resolution MSE, candidate-render counts, runtimes, and artifact hashes;
- aggregate means, per-target ratios, quantitative classification, plots, and deterministic blinded-review materials;
- an `.incomplete` lifecycle and preserved failure report;
- no learned-model path.

The runner passed `210 tests in 60.84 seconds` in Google Colab. The validation environment was Python 3.13.15 on Linux with NumPy 2.1.3, Pillow 11.3.0, and Matplotlib 3.10.0. Both primitive smoke planners were deterministic and monotonic. The explicit unauthorized probe raised `PermissionError` before creating a completed or incomplete output directory.

Exact evidence is preserved in:

- `configs/quadratic-bezier-runner-environment-2026-09-04.json`;
- `docs/quadratic-bezier-runner-pytest-2026-09-04.txt`;
- `docs/quadratic-bezier-runner-validation-2026-09-04.json`;
- `docs/quadratic-bezier-runner-unauthorized-probe-2026-09-04.txt`.

## One-time authorization and execution

Authorization commit `cc857407ed431c5583fd9e1c02a0ba619a8c187a` permits one completed execution only, using exact runner commit `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b` and output name `quadratic-bezier-fixed-comparison-v1`. An incomplete attempt must be preserved and reviewed before any recovery decision.

Use `notebooks/quadratic_bezier_fixed_comparison_execution.ipynb`. Run code cells 1–6 in order. The notebook mounts Google Drive, runs the complete tests again, validates the target freeze and authorization, checks for a fresh output path, executes the comparison with regular progress heartbeats, verifies every recorded artifact hash, and applies the blinded-review gate.

If blinded review is required, Cell 6 downloads only the blind handoff, montage, and blank review sheet. Do not run Cell 7 or inspect the mapping, numerical summaries, plots, or execution log until the blinded review is recorded. If blinded review is not required, Cell 6 downloads the numerical handoff directly.

## Boundaries

Do not interrupt, rerun, tune, replace targets, or open condition identities early. If an `.incomplete` directory appears, preserve it and stop. Do not commit generated binary outputs. No learned model is used in this comparison.
