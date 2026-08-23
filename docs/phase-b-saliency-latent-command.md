# Phase B0 validation command

## Status

Phase B0 development, formal B0, saliency scheduling B1, and RGB/high-resolution B2 are unauthorized. This command validates only the fixed architecture and objectives with deterministic random dummy tensors.

## Checkout

```bash
git fetch origin
git switch phase-b/saliency-latent
git pull --ff-only
```

## Full regression suite

```bash
pytest
```

Expected after this implementation: `111 passed`.

## Guarded validation-only runner

```bash
python experiments/20_phase_b_joint_embedding.py --validate-only
```

Expected status:

```text
phase_b0_architecture_and_objectives_valid_unauthorized
```

The result should report 392,345 trainable parameters, both latent shapes, finite losses and gradients, a frozen target encoder, zero EMA implementation error, and no model/data/output side effects.

## Prohibited commands

There is no authorized development command. Do not generate transitions, targets, planner states, candidate sets, checkpoints, or outputs. Do not add a training flag manually. Development may be authorized only after the complete test suite and validation-only runner pass and the implementation manifest is archived in a separate commit.
