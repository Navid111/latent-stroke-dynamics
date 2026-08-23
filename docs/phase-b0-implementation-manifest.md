# Phase B0 implementation manifest

## Lifecycle

- Validation date: 2026-08-23 (Asia/Dhaka)
- Branch: `phase-b/saliency-latent`
- Status: architecture and objectives validated; development unauthorized
- Formal Phase B0: unauthorized
- Phase B1 saliency scheduling: unauthorized
- Phase B2 RGB/high-resolution painter: unauthorized
- Experimental renderer data generated during validation: no
- Model trained on renderer data during validation: no

This manifest archives implementation evidence only. The reported losses came from deterministic random dummy tensors and are not scientific results, baselines, or evidence of model quality.

## Frozen references

- Closed Stage A base: `c211c3ab3a37b9c37eda5ba3c07c01173fd4c7f7`
- Phase B0 protocol freeze: `b3c2c284741dccb413b7811633336edc5e548b26`
- Validation scaffold: `aa92a85cf3150c59a952430ebb613e38b34d5703`
- Deterministic EMA arithmetic correction: `f57eecb06737950912c738e60ebc40ee3ec8dfc2`
- Frozen config path: `configs/phase-b-saliency-latent-2026-08-23.json`
- Frozen config blob SHA: `1c1862372d82a3f202daa93a4ceb94d40b4d7397`
- Implementation path: `src/latent_stroke_dynamics/phase_b_joint_embedding.py`
- Implementation blob SHA after correction: `c705de6ab61a1c90cf6969fe9b3632d579fd3ac6`

The correction commit changed one line only: it evaluated the same EMA equation through the same deterministic arithmetic path as the reference assertion. It did not change the EMA momentum, architecture, objective, data, or scientific protocol.

## Exact architecture inventory

- Model: `MultiScaleActionJointEmbeddingModel`
- Trainable parameters: **392,345**
- Frozen cap: **500,000**
- Cap margin: **107,655**
- Input: `[batch, 1, 64, 64]`
- Online latent scale 32: `[batch, 32, 32, 32]`
- Online latent scale 16: `[batch, 64, 16, 16]`
- Action scale 32: `[batch, 16, 32, 32]`
- Action scale 16: `[batch, 32, 16, 16]`
- Residual scale 32: `[batch, 32, 32, 32]`
- Residual scale 16: `[batch, 64, 16, 16]`
- Progress output: `[batch]`
- Target encoder: exact online-encoder copy, frozen to gradients, EMA momentum `0.99`
- Decoder: none

## Local validation evidence

Navid ran the complete suite locally on:

- macOS / Darwin
- Python `3.14.4`
- pytest `9.1.1`

Result:

```text
111 passed
```

Validation runner status:

```text
phase_b0_architecture_and_objectives_valid_unauthorized
```

Integrity checks:

- all trainable gradients present: passed
- all trainable gradients finite: passed
- target encoder frozen: passed
- EMA maximum error: `0.0`
- parameter cap: passed
- dummy tensors only: passed
- historical checkpoints loaded: no
- renderer transitions generated: no
- targets generated: no
- state banks generated: no
- candidate sets generated: no
- output directories created: no
- models trained on renderer data: no

## Dummy-only objective values

These values verify finite execution only and must not be interpreted as model performance:

| Component | Value |
|---|---:|
| Joint prediction, scale 32 | 0.0657934695482254 |
| Joint prediction, scale 16 | 0.0741327777504921 |
| Joint prediction, combined | 0.06996312737464905 |
| Variance penalty | 0.712358832359314 |
| Covariance penalty | 0.011149128898978233 |
| No-op consistency | 0.06362631916999817 |
| Progress regression | 0.0009532165713608265 |
| Candidate ranking | 1.4147374629974365 |
| Total | 0.5825915336608887 |

## Gate decision

The architecture and objective implementation gate **passed**. This permits implementation of a guarded development runner, but it does not itself authorize renderer-data generation or training.

Before one-time development authorization:

1. implement the exact frozen data, training, evaluation, artifact, and lifecycle guards;
2. keep development authorization false;
3. run the complete test suite and runner validation with no data/output side effects;
4. archive the validated runner state;
5. issue a separate authorization commit.
