# Quadratic-Bezier extension implementation gate

This branch begins with a validation-only scaffold. It adds:

- deterministic quadratic-Bezier rasterization;
- curve masks and analytically fitted RGB colors;
- unique changing curve proposals;
- matched exact-pixel planning for straight and curved primitives;
- action serialization round trips;
- deterministic common-resolution high-resolution replay;
- six deterministic procedural rights-safe targets;
- a locked validation config and decision rule;
- a no-output validation command and focused tests.

The scaffold deliberately omits an execution CLI. `execution_authorized` and `target_hashes_frozen` remain false in the config. The six target hashes printed by validation must be reviewed and frozen in a separate commit before a guarded comparison runner can be added or enabled.

## Recommended Colab handoff

Open `notebooks/quadratic_bezier_extension_validation.ipynb` from the public `quadratic-bezier-extension` branch and run all five cells on a standard CPU runtime. The notebook needs no token, no upload, no GPU, and no Drive mount. It pins exact implementation commit `7bdcd2e847ca7c5a1faf8a086b26441d8de1a4e1`, runs the complete test suite, validates the no-output boundary, prints the six proposed target hashes, and downloads the test log plus validation JSON.

## Equivalent local handoff

```bash
git switch quadratic-bezier-extension
git pull --ff-only
python -m pytest
python validate_quadratic_bezier_extension.py --validate-only
```

Expected validation status:

```text
quadratic_bezier_extension_valid_no_outputs
```

The report must show both primitive smoke planners as deterministic and monotonic, zero output side effects, no learned model, no training, no comparative outputs viewed, and no closed experiment changes.
