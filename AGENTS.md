# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/thesis-plan.md`;
3. frozen Gate 2, pixel-control, and Stage 3 protocols/results;
4. `docs/representation-extension-protocol.md`;
5. `docs/representation-extension-full-results.md`;
6. relevant source files and tests.

## Frozen completed evidence

- Gate 1 passed.
- DINOv2 Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval.
- Controlled Stage 3 succeeded across six targets.
- MNIST qualitative work documented learned long-horizon degradation.
- The single full representation extension completed on untouched seeds.

Do not rerun, retune, relabel, or replace these results.

## Active task

Validate the pure saved-summary adjudicator:

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/11_representation_extension_adjudication.py
```

This is not an experiment rerun. It reads the completed JSON only and saves a derived protocol interpretation.

## Extension interpretation boundary

- Preserve the raw summary unchanged.
- The written protocol requires 100% exact-target oracle retrieval and unique encoded candidates; it did not require bit equality between separately batched candidate-zero encodings.
- The written at-or-below-35% retrieval rule takes precedence over the average-error category.
- Pending local validation, task autoencoder is average-predictable but not action-usable; ViT-MAE is not predictively usable.
- Historical decisions remain unchanged.

## Prohibitions

- Never run experiment 10 again.
- No scientific setting changes, retraining, metric recomputation, or test tuning.
- No additional encoders, fine-tuning, joint training, latent planning, or prior-result reruns before thesis integration.
- Preserve all negative outcomes and limitations.

## Code standards

- Use fixed seeds and saved configs.
- Keep data, representation, dynamics, retrieval, and evaluation separate.
- Never silently catch experiment-invalidating errors.
- Do not commit credentials, model weights, caches, generated datasets, or raw uncompressed outputs.
