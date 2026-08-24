# Phase B0 manifest-only compatibility handoff

## Local validation

From the repository on branch `phase-b/saliency-latent`:

```bash
git pull --ff-only
source .venv/bin/activate
python -m pytest -q
```

Expected after this implementation: `153 passed`.

Then run the non-training local compatibility check into a new directory outside the repository:

```bash
LOCAL_COMPAT="$HOME/Desktop/phase-b0-manifest-compatibility-local-2026-08-24"
python experiments/27_phase_b_colab_manifest_compatibility.py \
  --output-dir "$LOCAL_COMPAT"
```

The command loads no model resources and cannot train. It should end with:

```text
All four original hashes matched. Training remains unauthorized.
```

Do not delete or rerun into the same directory. Preserve its report and manifests.

## Bundle creation after a passing local result

```bash
python scripts/build_phase_b_colab_manifest_compatibility_bundle.py \
  --local-report "$LOCAL_COMPAT/manifest_compatibility_report.json" \
  --local-test-count 153
```

The builder creates one resource-free archive under `dist/` and prints its SHA-256. It refuses a failed local hash gate, the wrong test count, tracked changes, or the wrong branch.

## Colab boundary

Use a fresh standard CPU Colab runtime and the notebook:

```text
notebooks/phase_b0_colab_manifest_compatibility.ipynb
```

Upload only the newly built compatibility bundle. The notebook creates an isolated environment pinned to NumPy `2.5.2`, Pillow `12.3.0`, and pytest `9.1.1`, while requiring PyTorch base version `2.11.0`.

It does not mount Drive. It downloads exactly five evidence files: one report and four manifests. Send those files and the eight-test output for review. Do not start training even if all hashes match.
