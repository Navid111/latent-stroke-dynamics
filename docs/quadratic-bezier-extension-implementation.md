# Quadratic-Bezier extension implementation gate

This branch begins with a validation-only scaffold. It adds:

- deterministic quadratic-Bezier rasterization;
- curve masks and analytically fitted RGB colors;
- unique changing curve proposals;
- matched exact-pixel planning for straight and curved primitives;
- action serialization round trips;
- six deterministic procedural rights-safe targets;
- a locked validation config and decision rule;
- a no-output validation command and focused tests.

The scaffold deliberately omits an execution CLI. `execution_authorized` and `target_hashes_frozen` remain false in the config. The six target hashes printed by validation must be reviewed and frozen in a separate commit before a guarded comparison runner can be added or enabled.

## Local handoff

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
