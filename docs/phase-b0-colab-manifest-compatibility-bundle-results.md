# Phase B0 Colab manifest compatibility — bundle handoff

## Bundle

The resource-free compatibility bundle was built successfully from the passing local hash gate.

```text
status: phase_b0_colab_manifest_compatibility_bundle_created_unauthorized
path: dist/phase-b0-colab-manifest-compatibility-b2a8cc690ae9.tar.gz
SHA-256: b890d1fad0501971964b4bcc743f3d47e39e800ba33cfebbde76eac103f21856
size: 1,870,860 bytes
source commit: b2a8cc690ae93c7eca1b35f88a14a34f39a8d58c
resources: 0
local tests: 153
Colab boundary tests: 8
scientific training allowed: false
recovery authorized: false
```

The local compatibility report included by reference has SHA-256:

```text
6c3b331ab6743b20f28de2ab752a07d695a3059c2c63c8ddc6baed4c624c6f63
```

## Contents and boundary

The archive contains:

- `bundle_manifest.json`; and
- `repository.bundle`, an internally verified Git bundle for source commit `b2a8cc690ae93c7eca1b35f88a14a34f39a8d58c`.

It contains no model resources, historical incomplete output, or recovery output. The internal `repository.bundle` does not need to be separately uploaded or inspected; it must remain inside the tar archive and is restored by the fail-closed Colab notebook.

## Next action

Use a fresh standard CPU Colab runtime and `notebooks/phase_b0_colab_manifest_compatibility.ipynb`. Upload the complete tar archive when prompted. The notebook must pass eight boundary tests, generate exactly four manifests under ephemeral `/content`, and download one report plus the four manifests. It does not mount Google Drive and cannot train.
