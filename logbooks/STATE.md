# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Frozen post-core representation extension  
**Status:** Protocol/config committed before implementation and data generation

## Frozen completed evidence

- Gate 1 passed.
- DINOv2 Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval.
- Controlled Stage 3 learned planning succeeded across six synthetic targets.
- MNIST qualitative exact greedy reached MSE `0.022196`; learned reached `0.040860` and degraded after step 33.
- User-facing painter and best-painting output passed all 38 tests.

No completed result may be rerun, retuned, or relabeled.

## Active extension

Frozen files:

- `docs/representation-extension-protocol.md`;
- `configs/representation-extension-2026-08-22.json`.

New representations:

1. frozen deterministic unmasked final spatial tokens from `facebook/vit-mae-base`;
2. a small reconstruction-trained 32×16×16 convolutional autoencoder latent.

Historical DINOv2 and pixel outcomes are anchors only and will not be rerun. The extension uses untouched seeds `20261024`–`20261030`; development smoke uses `20261020`–`20261022`.

Each new representation is classified independently by average residual error, crowding behavior, four-way action retrieval, and integrity checks. The required action-usable retrieval threshold remains 50%.

## Immediate next action

Implement, without generating extension data:

1. deterministic unmasked ViT-MAE wrapper and repeatability test;
2. convolutional autoencoder and reconstruction tests;
3. latent standardization and checkpoint-integrity helpers;
4. shared extension configuration validation;
5. smoke/full-run guards.

Then run the complete repository tests before any development smoke.

## Boundaries

- No new extension data have been generated.
- No test split may enter fitting, normalization, or selection.
- Do not substitute a different pretrained encoder if ViT-MAE deterministic unmasked extraction fails.
- No additional encoder, joint training, contrastive loss, or latent planner before both frozen representations are archived.
