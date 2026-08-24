# Phase B0 manifest-only compatibility protocol

## Purpose

This protocol tests whether the original Mac Phase B0 data manifests can be reproduced byte-for-byte in a fresh Linux/Colab environment when the renderer-related dependencies are pinned to the Mac versions.

It exists because the consumed recovery attempt stopped before training when all four Colab-generated manifest hashes differed. The forensic comparison found sparse differences only in non-no-op rendered strokes, strongly implicating the Pillow line-rasterization boundary.

## Frozen hypothesis

A Linux CPU environment using:

- NumPy `2.5.2`
- Pillow `12.3.0`
- PyTorch base version `2.11.0`
- the unchanged `PIL.ImageDraw.line` renderer

may reproduce all four original manifest hashes exactly.

## Allowed work

The validator may generate, in memory:

- the frozen train, validation, and diagnostic transition splits;
- the frozen planner targets, states, candidate sets, and progress labels; and
- the same four JSON manifests used by the original continuity gate.

It persists only those four JSON manifests and one compatibility report. It runs sequentially on CPU to reduce memory pressure.

## Forbidden work

The validator cannot:

- load the autoencoder, latent predictors, or pixel predictor;
- create model checkpoints, training histories, target images, decisions, or recovery outputs;
- call the recovery runner or reuse its consumed authorization;
- write inside the repository, Google Drive, or either historical `.incomplete` directory;
- train any scientific model; or
- authorize Phase B0 recovery, formal Phase B0, B1, or B2.

## Exact gate

The check passes only if the generated directory contains exactly these four files and every SHA-256 matches the original Mac manifest:

| Manifest | Required SHA-256 |
|---|---|
| `diagnostic_test_transitions.json` | `d3101d29de97659a44932282fcbeed807405eecc1f678e71fd36e96a600d997a` |
| `planner_supervision.json` | `02bef6101b0e380651301bbf7c8c0cf5e02c7c2a39e2dbab13e44fac1a9d186a` |
| `train_transitions.json` | `7bb572b4d053649d22de75584615441b9d72c014f1a6128b435677e560c6304b` |
| `validation_transitions.json` | `234ff3b68399aea160ceb0665728d9f1d3d5971e1924ab36eef5c1537558c817` |

A mismatch is preserved as useful infrastructure evidence and does not become a scientific failure.

## Interpretation

### If all four hashes match

The pinned environment is compatible with the original data boundary. The result must be archived first. A separate recovery protocol, fresh output root, fresh validation, and fresh one-time authorization are still required before training.

### If any hash differs

Do not accept the Linux hashes and do not train. Preserve the report and four generated manifests. The next protocol must either package immutable Mac-generated data/tensors or replace the renderer with a separately validated platform-independent boundary.
