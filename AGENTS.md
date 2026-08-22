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
6. `docs/representation-extension-final-decision.md`;
7. relevant source files and tests.

## Frozen completed evidence

- Gate 1 passed.
- DINOv2 Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval.
- Controlled Stage 3 succeeded across six targets.
- MNIST qualitative work documented learned long-horizon degradation.
- The single full representation extension completed and its written-protocol adjudication passed all 54 tests.

Do not rerun, retune, relabel, or replace these results.

## Current phase

Experimental work is frozen. The active phase is thesis integration:

- update Methods, Results, Discussion, and Limitations;
- select final tables and figures;
- verify literature citations against original PDFs;
- prepare the reproducibility instructions and defence narrative.

Small reporting, documentation, and packaging fixes are allowed. New scientific runs require an explicit new protocol and must not replace existing evidence.

## Final extension outcomes

- task autoencoder: average-predictable but not action-usable, 37.89% retrieval;
- frozen ViT-MAE: not predictively usable, 7.11% retrieval;
- raw pixels remain the strongest tested action representation, 100% retrieval;
- historical decisions remain unchanged.

## Prohibitions

- Never run experiment 10 again.
- Never overwrite the raw extension summary or final adjudication.
- No post-test tuning, result substitution, or hidden negative outcomes.
- No extra encoder, fine-tuning, joint training, latent planning, RL, color, or textured-brush scope before thesis integration is complete.
- Do not commit credentials, model weights, caches, generated datasets, or raw uncompressed outputs.
