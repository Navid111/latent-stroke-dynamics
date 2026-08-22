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

The single full representation-extension command is committed but not yet authorized to generate data. The only next actions are local tests and `--validate-only`.

Authorized now:

- `pytest`;
- `python experiments/10_representation_extension_full.py --validate-only`;
- implementation repair only if either validation fails.

Not authorized until validation is reviewed:

- `--run-frozen-extension`;
- generating seeds `20261024`–`20261030`;
- changing any scientific setting;
- additional encoders, fine-tuning, joint training, or latent planning;
- any prior-result rerun.

## Interpretation boundaries

- Development metrics are diagnostic only.
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
