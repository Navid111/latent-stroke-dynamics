# AGENTS.md

This file is the operating contract for coding and research agents working in this repository.

## Project purpose

This is a bachelor’s thesis project on action-conditioned canvas dynamics and sequential stroke-based rendering, due 2026-09-24.

## Required reading order

Before changing code or proposing work, read:

1. `logbooks/STATE.md`;
2. `docs/thesis-plan.md`;
3. `docs/gate-2-protocol.md` and `docs/gate-2-results.md`;
4. `docs/pixel-space-control-protocol.md` and `docs/pixel-control-results.md`;
5. `docs/stage-3-pixel-planner-protocol.md` and `docs/stage-3-controlled-results.md`;
6. `docs/representation-extension-protocol.md` and current extension handoff;
7. relevant source files and tests.

## Frozen completed results

- Gate 1 passed.
- Latent Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval.
- Controlled Stage 3 succeeded across six targets.
- Qualitative MNIST documented learned long-horizon degradation.
- Representation development smoke completed with integrity and no primary/stress data.

Do not rerun, retune, relabel, or replace these results.

## Active scope

The full representation-extension command passed 51 tests and validation-only review. Exactly one full run is authorized using:

```bash
python experiments/10_representation_extension_full.py --run-frozen-extension
```

No flags or concurrent process are authorized.

## Run boundaries

- Execute the frozen command once.
- Do not edit code, config, architecture, seeds, thresholds, epochs, or output paths before or during the run.
- On success, never rerun or retune; preserve all results.
- On error/interruption, preserve the traceback and `.incomplete` directory and request review before any retry.
- No additional encoders, fine-tuning, joint training, latent planning, or prior-result reruns.

## Interpretation boundaries

- The task autoencoder is reconstruction-trained, not JEPA.
- ViT-MAE and the task autoencoder are two formulations, not an exhaustive benchmark.
- Historical DINOv2 and pixels remain descriptive anchors.
- No extension outcome changes a prior frozen decision.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Code and experiment standards

- Keep data, representation, dynamics, retrieval, and evaluation separate.
- Use fixed seeds, matched budgets, saved configs, and no silent overwrite.
- Never silently catch experiment-invalidating errors.
- Do not commit credentials, model weights, caches, generated datasets, or raw outputs.
- Preserve negative outcomes and limitations.

## Handoff protocol

After meaningful work, test, update state/logbook, and archive compact results without rewriting prior evidence.
