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

The ranking-aware latent protocol/config and validation foundation are committed. No follow-up data are authorized.

Current next action is local validation only:

```bash
pytest
python experiments/12_ranking_aware_latent_followup.py --validate-only
```

The validation result must expose the existing latent-statistics SHA-256. Freeze that hash in a separate commit before implementing or authorizing development.

## Hard boundaries

- Do not call any transition generator for this follow-up yet.
- Do not create either follow-up output directory.
- Do not generate formal seeds `20261104`–`20261110`.
- Do not run experiment 10 again.
- Do not retrain or fine-tune the frozen task autoencoder.
- Do not change canvas, renderer, stroke family, architecture, ranking grid, thresholds, or formal sizes.
- Do not use diagnostic-test or formal-test results for selection.

## Code standards

Keep follow-up code separate from completed experiment code. Validate hashes before data generation. Use atomic output handling later. Preserve negative outcomes and never commit credentials, checkpoints, caches, generated datasets, or raw output directories.
