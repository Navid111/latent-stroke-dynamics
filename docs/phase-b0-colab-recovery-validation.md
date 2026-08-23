# Phase B0 guarded recovery runner — CUDA validation

## Scope

This is a second, dummy-only cloud gate for the modified recovery code. It is distinct from the earlier general CUDA preflight. It exercises device-aware one-epoch dummy fits for both frozen variants, GPU diagnostics, GPU candidate scoring, temporary checkpoint save/hash verification, exact resource hashes, loaded-state hashes, and the recovery lifecycle boundary.

It does not generate renderer transitions, targets, state banks, planner candidate sets, recovery output, or scientific checkpoints. Temporary dummy checkpoints are deleted before the report is returned. Dummy losses, hashes, and scores are implementation evidence only.

## Local packaging handoff

From the repository root with the virtual environment active:

```bash
git pull --ff-only
python -m pytest
python experiments/23_phase_b_colab_recovery.py --validate-only
python scripts/build_phase_b_colab_recovery_validation_bundle.py
```

Expected local suite: `138 passed`.

The builder prints a bundle path, SHA-256, exact source commit, six-resource count, and `recovery_authorized: false`.

## Colab procedure

1. Open `notebooks/phase_b0_colab_recovery_validation.ipynb` from the private GitHub branch.
2. Select a Tesla T4 GPU runtime.
3. Run cells in order.
4. Upload exactly the newly generated recovery-validation bundle.
5. Expect `138 passed` in Colab.
6. Download `phase-b0-colab-recovery-validation-report.json`.
7. Delete the runtime after the report is safely downloaded.

Expected report status:

```txt
phase_b0_colab_recovery_implementation_valid_unauthorized
```

Do not mount Google Drive and do not execute the real recovery command during this validation. A passing report still requires archival and a separate one-time authorization commit before any scientific execution.
