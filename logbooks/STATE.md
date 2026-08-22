# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Frozen post-core representation extension  
**Status:** Foundation implemented; local validation pending

## Frozen completed evidence

- Gate 1 passed.
- DINOv2 Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval.
- Controlled Stage 3 learned planning succeeded across six synthetic targets.
- MNIST qualitative exact greedy reached MSE `0.022196`; learned reached `0.040860` and degraded after step 33.
- User-facing painter and best-painting output passed all 38 tests.

No completed result may be rerun, retuned, or relabeled.

## Active extension

Frozen before implementation:

- `docs/representation-extension-protocol.md`;
- `configs/representation-extension-2026-08-22.json`.

Implemented without extension-data generation:

- deterministic unmasked raster-ordered ViT-MAE wrapper;
- fixed MAE noise and restoration helpers;
- frozen 32×16×16 convolutional autoencoder architecture;
- train-only latent channel standardization;
- reconstruction and freeze helpers;
- configuration-drift and seed-overlap guards;
- eight unit tests;
- validation-only and two-image MAE smoke commands.

## Immediate next action

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/08_representation_extension.py --validate-only
python experiments/08_representation_extension.py --mae-smoke
```

Expected: `46 passed`, then `foundation_valid`, then `mae_encoder_smoke_passed`. The MAE smoke may download model weights but generates no extension split.

## Boundaries

- Do not run any full extension command yet.
- No extension split has been generated.
- No test split may enter fitting, normalization, or selection.
- Do not substitute another pretrained encoder if the deterministic MAE smoke fails.
- No additional encoder, joint training, contrastive loss, or latent planner before both frozen representations are archived.
