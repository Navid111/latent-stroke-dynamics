# Phase B0 Google Colab CUDA preflight

## Scope

This preflight is infrastructure-only. It may use deterministic random dummy tensors and load frozen historical files solely to verify hashes. It may not generate renderer transitions, targets, state banks, candidate sets, scientific checkpoints, or scientific results. Phase B0 recovery remains unauthorized.

## Local preparation

From the repository root on the Mac:

```bash
git pull --ff-only
pytest
python scripts/build_phase_b_colab_preflight_bundle.py
```

Expected test count: `120 passed`.

The bundle command prints a path under `dist/` and its SHA-256. The bundle contains:

- a Git bundle of the exact current branch;
- the six frozen historical resources actually used by Phase B0;
- a manifest recording the branch, source commit, and raw hashes.

It excludes the preserved local `.incomplete` directory and all Phase B scientific outputs.

## Colab

1. In Colab, select **Runtime → Change runtime type → T4 GPU** or another available GPU.
2. Open `notebooks/phase_b0_colab_preflight.ipynb` from the private GitHub repository.
3. Run cells in order.
4. Upload the single `dist/phase-b0-colab-preflight-<commit>.tar.gz` file when prompted.
5. Download and send the final `phase-b0-colab-preflight-report.json`.

Passing status:

```text
phase_b0_colab_cuda_preflight_passed_recovery_unauthorized
```

## Interpretation

Dummy losses and benchmark timing are implementation checks, not scientific evidence. A pass does not authorize recovery. A separate hardware-recovery protocol, runner validation, and one-time authorization are required before real Phase B data generation or training.
