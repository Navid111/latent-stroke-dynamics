# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Frozen post-core representation extension  
**Status:** Development smoke archived; full command not yet frozen

## Frozen completed evidence

- Gate 1 passed.
- DINOv2 Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval.
- Controlled Stage 3 learned planning succeeded across six synthetic targets.
- MNIST qualitative exact greedy reached MSE `0.022196`; learned reached `0.040860` and degraded after step 33.

No completed result may be rerun, retuned, or relabeled.

## Development smoke

- Status: complete and non-decision-making.
- Runtime: 173.49 seconds.
- Integrity: passed.
- Primary/stress data: not generated.

Task-autoencoder dynamics showed 71.01% action-region improvement versus identity and 46.35% four-way retrieval. ViT-MAE showed 23.97% improvement and 7.29% retrieval. These values cannot classify either representation.

The development autoencoder did not satisfy the reconstruction threshold: it was 14.02% worse than the train-mean-image validation baseline. Its latent was non-collapsed and checkpoint reload was exact. Frozen settings remain unchanged for the full run.

The raw summary's autoencoder `parameter_count: 0` was a reporting-only bug caused by counting trainable parameters after freezing. The architecture has 49,569 total parameters. The helper is corrected prospectively; the raw artifact remains preserved.

## Immediate next action

Implement and separately commit the single guarded full command using untouched seeds `20261024`–`20261030`. Run tests and validation-only mode before authorizing primary/stress generation.

## Boundaries

- Do not rerun the development smoke.
- Do not change architecture, losses, thresholds, epochs, or seeds from smoke metrics.
- Do not generate primary/stress data until the full command is committed and validation-only checks pass.
- No additional encoder, joint training, contrastive loss, or latent planner before both frozen representations are archived.
