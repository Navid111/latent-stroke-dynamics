# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

Bachelor's thesis on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

1. `logbooks/STATE.md`;
2. frozen core and representation-extension results;
3. `docs/ranking-aware-latent-protocol.md`;
4. `docs/ranking-aware-latent-handoff.md`;
5. relevant source files and tests.

## Frozen completed evidence

Gate 1, DINOv2 Gate 2, paired pixel control, controlled Stage 3, qualitative MNIST, and the full representation extension are complete. Never rerun, retune, relabel, or replace them.

## Active study

The ranking-aware latent protocol, task-autoencoder state hash, and latent-statistics file hash are frozen. No follow-up data are authorized.

Current work is to implement and test a guarded development-grid runner. It must support validation-only mode before development authorization and must keep formal seeds untouched.

## Hard boundaries

- Do not generate follow-up development data until the guarded runner passes local validation and a separate authorization commit lands.
- Do not create either follow-up output directory during validation.
- Do not generate formal seeds `20261104`–`20261110`.
- Do not run experiment 10 again.
- Do not retrain or fine-tune the frozen task autoencoder.
- Do not change canvas, renderer, stroke family, architecture, ranking grid, thresholds, or formal sizes.
- Do not use diagnostic-test or formal-test results for selection.

## Code standards

Keep follow-up code separate from completed experiment code. Validate hashes before data generation. Use atomic `.incomplete` output handling and overwrite refusal. Preserve negative outcomes and never commit credentials, checkpoints, caches, generated datasets, or raw output directories.
