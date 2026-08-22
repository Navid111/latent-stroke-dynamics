# Post-core representation extension protocol — 2026-08-22

**Status:** Frozen before implementation and before any extension data generation  
**Role:** Explanatory post-core study  
**Deadline:** 2026-09-24  
**Original Gate 2 decision:** Remains fail  
**Pixel control and Stage 3 decisions:** Remain unchanged

## 1. Motivation and question

The completed DINOv2 Gate 2 established that one frozen patch-token formulation was predictable under average error but insufficient for exact action ranking. The paired pixel control established that the deterministic transition and action identity were recoverable at full pixel resolution. Those results do not establish whether the failure generalizes to other frozen encoders or whether a representation trained on the project's own canvases can preserve stroke-level precision.

This extension asks two narrow questions:

1. Does a frozen reconstruction-oriented pretrained encoder preserve enough one-stroke information for action-conditioned next-latent prediction and four-way action retrieval?
2. Does a small spatial autoencoder trained only on renderer-generated canvases produce a latent space that supports the same task?

The extension does not claim a new JEPA architecture. It does not fine-tune DINOv2, revise Gate 2, or use qualitative images for training.

## 2. Provenance and evidential status

This protocol was written after observing:

- DINOv2 Gate 2 retrieval of 27.7%;
- paired pixel retrieval of 100%;
- successful controlled pixel planning;
- long-horizon learned-pixel degradation on the MNIST qualitative trajectory.

It is therefore a preregistered post-core extension, not an independent confirmation of the original hypotheses. New test seeds are untouched at freeze time. Both success and failure must be preserved.

## 3. Frozen representation ladder

### Historical anchors — no rerun

1. **Frozen DINOv2-small spatial tokens:** use the already-recorded Gate 2 result as the semantic/self-distillation anchor.
2. **Raw normalized pixels:** use the already-recorded paired pixel control as the full-resolution anchor.

Historical values are descriptive anchors because they used earlier paired seeds. They are not recomputed, relabeled, or included in the new extension decision.

### New representation A — frozen ViT-MAE

- Model: `facebook/vit-mae-base`.
- Weights: frozen throughout.
- Input: Pillow grayscale canvas converted to RGB through the model's official image processor.
- Resolution: the processor's native 224×224 input.
- Spatial state: deterministic unmasked final encoder patch tokens; no class token.
- Expected patch grid: 14×14 from 16×16 patches.
- Feature dimension: 768.
- Per-token representation: L2-normalized before residual construction.
- Device: CPU.
- Encoding batch size: 8, reducible only for memory without changing values.

The implementation must explicitly disable random masking. Repeated encodings of the same image batch must agree within maximum absolute difference `1e-7`. If deterministic unmasked spatial tokens cannot be obtained from the installed model API, stop as implementation-invalid before generating extension data; do not silently substitute another model.

### New representation B — task-trained convolutional autoencoder

Train a small autoencoder from scratch on generated train canvases only, then freeze its encoder before dynamics training.

Encoder:

```text
1×64×64
→ Conv 1→16, 3×3, stride 1, padding 1; GELU
→ Conv 16→32, 4×4, stride 2, padding 1; GELU
→ Conv 32→32, 4×4, stride 2, padding 1; GELU
→ 32×16×16 latent
```

Decoder:

```text
32×16×16
→ ConvTranspose 32→32, 4×4, stride 2, padding 1; GELU
→ ConvTranspose 32→16, 4×4, stride 2, padding 1; GELU
→ Conv 16→1, 3×3, stride 1, padding 1; sigmoid
→ 1×64×64 reconstruction
```

The dynamics representation is the 16×16 grid of 32-dimensional encoder vectors. Channels are standardized with training-set latent means and standard deviations computed after autoencoder selection; no validation or test statistics enter the transform.

This is a reconstruction-trained task-specific latent, not JEPA. No transition labels enter autoencoder training.

## 4. Frozen data

Use the existing deterministic transition generator and distribution, but new untouched seeds.

### Primary splits

| Split | Examples | Seed | Use |
|---|---:|---:|---|
| Train | 1,000 | `20261024` | Autoencoder and dynamics fitting |
| Validation | 200 | `20261025` | Early stopping and model-family selection |
| Test | 300 | `20261026` | One final extension evaluation |

Primary crowding is `{0, 5, 15}`, widths are `{1, 2, 3, 4}`, values are `{0, 32, 64, 96, 128}`, and minimum normalized length is `0.20`.

Autoencoder image training uses the union of current and next canvases from the train transitions only. Validation reconstruction uses the corresponding validation canvases. Test canvases and counterfactuals are not used for autoencoder fitting or selection.

### Secondary stress slices

These cannot change the primary classification.

| Slice | Examples | Seed |
|---|---:|---:|
| Unseen width 5 | 100 | `20261027` |
| Unseen intensities `{16, 80, 176}` | 100 | `20261028` |
| High crowding 30 | 100 | `20261029` |
| High crowding 60 | 100 | `20261030` |

### Development-only smoke

Implementation smoke data use 128/32/64 transitions from seeds `20261020`, `20261021`, and `20261022`. Smoke outcomes cannot alter architecture, thresholds, formal split sizes, or test interpretation. Repairs are limited to implementation defects and M1 memory batch size.

## 5. Autoencoder training and selection

- Model seeds: `101`, `202`, `303`.
- Input pixels: normalized to `[0, 1]`.
- Objective: full-canvas reconstruction MSE.
- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.00001`.
- Batch size: 32.
- Maximum epochs: 50.
- Early-stopping patience: 8.
- Device: CPU.
- No image augmentation.

Select exactly one autoencoder seed using minimum validation reconstruction MSE. Test reconstruction metrics are evaluated once after selection.

Implementation validity requires:

1. finite train/validation losses;
2. selected validation reconstruction MSE below the validation mean-image baseline by at least 30%;
3. non-collapsed latent channels, defined as mean train-latent channel standard deviation above `1e-4`;
4. exact checkpoint reload reproducing identical encodings within `1e-7`.

A failure of these checks is an autoencoder implementation/result failure, not permission to redesign after test exposure.

## 6. Frozen action-conditioned dynamics

For each new representation:

```text
z_current = E(C_current)
z_next    = E(C_next)
delta     = z_next - z_current
z_hat_next = z_current + delta_hat(z_current, action)
```

Predictor inputs per spatial token:

1. current token;
2. existing seven-value global stroke vector;
3. fractional action-mask coverage at the representation's grid;
4. normalized token-center x/y coordinates.

Baselines and trainable families:

1. identity/no-change;
2. training-set mean delta;
3. shared linear patch predictor;
4. shared MLP patch predictor with hidden dimension 256 and at most one million trainable parameters.

Dynamics settings:

- model seeds `11`, `22`, `33`;
- AdamW;
- learning rate `0.001`;
- weight decay `0.0001`;
- batch size 16;
- maximum epochs 30;
- early-stopping patience 6;
- balanced residual loss: 50% action region, 50% outside region;
- CPU only.

For each representation, select linear or MLP by mean validation action-region MSE across its three seeds. Test retrieval is never used for selection.

## 7. Frozen retrieval diagnostic

For each held-out test transition, construct the same four pixel-distinct exact outcomes:

1. true stroke;
2. shifted position;
3. changed width;
4. changed intensity.

Encode all candidates in the evaluated representation. Score normalized-feature MSE over the union of candidate action regions. Candidate zero is true and chance is 25%.

Required diagnostics:

- four-way top-1 retrieval;
- candidate-selection frequencies;
- true-versus-position, true-versus-width, and true-versus-intensity win rates;
- mean true margin;
- exact-target oracle retrieval, which must equal 100%.

## 8. Metrics

### Primary per new representation

- action-region residual MSE;
- improvement versus identity and training mean delta;
- four-way top-1 retrieval;
- improvement versus identity at crowding 0, 5, and 15;
- seed spread and finite-metric checks.

### Secondary

- full/outside-region MSE;
- next-token cosine distance;
- factor-wise retrieval;
- results by width, intensity, length, and crowding;
- stress-slice performance;
- autoencoder reconstruction MSE and MAE;
- parameter counts, encoding time, training time, and peak-safe batch sizes;
- spatial residual/error figures.

Metrics across DINOv2, pixels, and this new split are compared descriptively because the data seeds differ.

## 9. Frozen representation classification

Classify each new representation independently.

### Action-usable

All conditions must pass:

1. selected predictor improves action-region MSE by at least 30% versus identity and mean delta;
2. improvement versus identity is positive at crowding 0, 5, and 15;
3. four-way retrieval is at least 50%;
4. oracle retrieval is 100%;
5. no collapse, leakage, duplicate-candidate, non-finite, or seed-instability failure occurs.

### Average-predictable but not action-usable

Average-error conditions pass, but retrieval is below 50% or one primary crowding level fails.

### Not predictively usable

Improvement is below 10% versus a trivial baseline, retrieval is at or below 35%, predictions ignore action, or implementation integrity fails.

These labels apply only to this extension. No outcome changes the recorded DINOv2 Gate 2 fail, paired pixel success, or Stage 3 success.

## 10. Interpretation matrix

| ViT-MAE | Task autoencoder | Interpretation |
|---|---|---|
| Action-usable | Action-usable | Reconstruction-preserving latents can support fine action ranking; DINOv2 failure is formulation-specific |
| Not usable | Action-usable | Task alignment is more important than generic frozen pretraining in this setup |
| Action-usable | Not usable | Reconstruction-oriented pretraining provides useful structure not recovered by the small autoencoder |
| Not usable | Not usable | The tested compressed latent formulations remain less action-precise than pixels; investigate resolution/loss/predictor interaction |

Do not infer universal encoder rankings from two new representations.

## 11. Required integrity checks

- protocol/config commit precedes extension data generation;
- all primary/stress split fingerprints are disjoint;
- no test row enters training, standardization, early stopping, or selection;
- MAE weights remain frozen and encodings deterministic;
- autoencoder is selected using validation reconstruction only;
- autoencoder encoder freezes before dynamics training;
- all counterfactual outcomes are pixel-distinct;
- all encoded candidates are checked for exact duplicates;
- all metrics are finite;
- tiny-overfit dynamics sanity check decreases loss;
- all seeds and model identifiers are saved;
- no qualitative MNIST data enter this experiment.

## 12. Required artifacts

Save:

- exact run config and environment metadata;
- split fingerprints;
- encoder/checkpoint metadata and hashes;
- autoencoder training histories and reconstruction summaries;
- train-only latent normalization statistics;
- dynamics training histories;
- per-example and aggregate metrics;
- retrieval and pairwise diagnostics;
- stress-slice tables;
- spatial/reconstruction figures;
- one final classification JSON.

Generated models, caches, and raw outputs remain ignored by Git. Compact final tables and figures may be archived under `results/representation-extension/`.

## 13. Execution order and stop rules

1. Commit this protocol and exact configuration.
2. Implement representation wrappers, autoencoder, integrity checks, and unit tests.
3. Run the complete repository suite.
4. Run deterministic encoder and tiny-overfit checks.
5. Run development-only smoke for both new representations.
6. Repair implementation defects only.
7. Freeze the single full command in a separate commit.
8. Run each new representation once on the primary extension splits.
9. Archive both outcomes without retuning.
10. Resume thesis writing.

Do not add additional pretrained encoders, joint encoder-dynamics training, contrastive losses, latent planning, RL, or architecture searches before this frozen comparison is complete.
