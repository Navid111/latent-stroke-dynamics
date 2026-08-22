# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. `docs/thesis-plan.md`;
3. frozen Gate 2, pixel-control, Stage 3, and representation-extension results;
4. `docs/representation-extension-final-decision.md`;
5. `docs/ranking-aware-latent-protocol.md`;
6. relevant source files and tests.

## Frozen completed evidence

- Gate 1 passed.
- DINOv2 Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval.
- Controlled Stage 3 succeeded across six targets.
- MNIST qualitative work documented learned long-horizon degradation.
- The full representation extension and written-protocol adjudication are complete.

Never rerun, retune, relabel, or replace completed results.

## Active study

A post-core ranking-aware latent follow-up is frozen before implementation and before data generation. It holds the successful task-autoencoder representation fixed and compares the same MLP under MSE-only and MSE-plus-counterfactual-ranking objectives.

Current next steps:

1. implement config validation, checkpoint/statistics validation-only mode, ranking loss, and unit tests;
2. record the existing latent-statistics SHA-256;
3. freeze that hash and authorize development only after validation;
4. keep all formal seeds untouched.

## Hard boundaries

- Do not generate follow-up development data yet.
- Do not generate reserved formal seeds `20261104`–`20261110`.
- Do not run experiment 10 again.
- Do not retrain or fine-tune the frozen task autoencoder.
- Do not change canvas size, renderer, stroke family, architecture, ranking grid, thresholds, or formal sizes.
- Do not use diagnostic-test or formal-test results for selection.
- Additional encoders, 128×128 canvases, new brushes, and true JEPA-inspired joint training require separate later protocols.

## Code standards

- Keep new follow-up code in separate modules/scripts so completed experiment code stays unchanged.
- Validate hashes and output paths before any data generation.
- Use atomic `.incomplete` output handling and refuse overwrite.
- Preserve negative outcomes and implementation failures.
- Do not commit credentials, model weights, caches, generated datasets, or raw outputs.
