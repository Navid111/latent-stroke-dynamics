# Phase B0 Colab recovery runner command

## Current lifecycle

The CUDA recovery runner is implemented but recovery remains unauthorized. Validation must not generate renderer data, targets, state banks, candidate sets, output directories, checkpoints, or scientific results.

## Local validation handoff

From the repository root with the virtual environment active:

```bash
git pull --ff-only
python -m pytest
python experiments/23_phase_b_colab_recovery.py --validate-only
```

Expected test count after the recovery-runner commits: `132 passed`.

Expected validation status:

```txt
phase_b0_colab_recovery_runner_valid_unauthorized
```

The validation JSON must report every scientific side-effect as false and `recovery_authorized: false`.

## Forbidden command

Do not run:

```bash
python experiments/23_phase_b_colab_recovery.py --recovery --artifact-root ...
```

The authorization guard is expected to reject it before output creation. Do not run the old `experiments/21_phase_b_development.py --development` command either.

## Implemented recovery behavior behind the guard

- exact Tesla T4 / Python / PyTorch / CUDA / cuDNN environment check;
- six frozen raw-resource and loaded-state checks;
- MSE-only predictor loading without unused ranking-aware checkpoints;
- deterministic regeneration and pre-training verification of all four archived manifest hashes;
- float32 CUDA training and diagnostics for the two frozen Phase B variants;
- immediate checkpoint and history persistence after each completed variant;
- CPU exact rendering and archived MSE-L1 planning;
- CUDA learned-pixel and Phase B planning;
- persistent Google Drive `.incomplete` directory;
- stage journal and partial long-horizon tables;
- six-hour fail-closed cap;
- final artifact hash manifest and atomic finalization;
- no automatic resume after interruption.

A passing local validation still does not authorize recovery. A new exact bundle and dummy-only Colab validation must pass before any separate one-time recovery authorization.
