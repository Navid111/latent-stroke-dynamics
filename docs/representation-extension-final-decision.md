# Final representation-extension decision — 2026-08-22

**Status:** Complete and archived  
**Single frozen run:** Completed once  
**Adjudication validation:** 54 tests passed in 6.02 seconds  
**Written-protocol global integrity:** Passed  
**Rerun or retuning:** Forbidden

## Audit trail

The raw extension summary remains unchanged. Its SHA-256 is:

```text
d5e53f277befc45ae55abb144f0391684885b343fb1eb6615b1ebe92d86c9c2b
```

A JSON-only adjudicator applied two written-protocol corrections without loading models or data, generating data, training, evaluating, or recomputing scientific metrics.

1. The frozen oracle requirement was 100% exact-target retrieval plus unique encoded candidates. The task autoencoder achieved both. Bit-identical separately batched candidate-zero encodings were not required.
2. The written at-or-below-35% retrieval rule takes precedence over the average-error label.

## Final classifications

| Representation | Improvement vs identity | Improvement vs mean delta | Retrieval | Final classification |
|---|---:|---:|---:|---|
| Task autoencoder | 70.65% | 68.62% | 37.89% | Average-predictable but not action-usable |
| Frozen ViT-MAE | 33.08% | 30.63% | 7.11% | Not predictively usable |

The task autoencoder passed reconstruction eligibility and every written integrity check. ViT-MAE also passed implementation integrity, but its retrieval was below the frozen not-usable boundary.

## Scientific interpretation

The task-trained spatial latent substantially improved upon the tested generic frozen latent formulations and modeled average stroke consequences well. It still did not preserve enough exact width/intensity action identity to reach the 50% action-usable requirement. Frozen ViT-MAE retained some position information but was especially weak on width and intensity ranking.

Raw pixels remain the strongest tested action representation at 100% retrieval. The representation ladder supports a restrained conclusion: task alignment improves latent transition predictability, but the tested compressed latents still lose planning-relevant stroke identity.

## Project consequence

- Keep the working painter pixel-based.
- Do not rerun or retune the extension.
- Do not revise any historical Gate 2, pixel-control, or Stage 3 decision.
- Move to thesis integration, figure/table selection, limitations, and defence preparation.
