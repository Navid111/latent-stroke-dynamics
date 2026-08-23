# Phase B0 guarded development runner

## Current lifecycle

The complete Phase B0 development implementation exists, but development remains unauthorized. The runner validation must not load historical checkpoints, generate renderer data, create output directories, or train a model.

## Sync

```bash
git pull --ff-only
```

## Full test suite

```bash
pytest
```

Expected: `116 passed`.

## Unauthorized runner validation

```bash
python experiments/21_phase_b_development.py --validate-only
```

Expected status:

```text
phase_b0_development_runner_valid_unauthorized
```

The report must say that development, formal B0, B1, and B2 are unauthorized and that no models, renderer transitions, targets, state banks, candidates, outputs, or training were created.

## Prohibited

Do not run `--development`. The config authorization is false, so the command will stop before model loading, data generation, or output creation. A separate commit is required to authorize exactly one development execution after the full suite and validation report pass.
