# Latent Stroke Dynamics

Bachelor's thesis experiments on **action-conditioned latent canvas dynamics for stroke-based rendering**.

## Current status

- **Gate 1 passed** on 2026-08-19.
- **Latent Gate 2 formally failed** on 2026-08-21 because retrieval was 27.7%.
- **Pixel-control smoke passed all engineering checks** on 2026-08-21.
- **The single frozen paired pixel-control run is authorized.**

## Why the pixel control exists

The latent model predicted broad one-step consequences well but confused exact stroke width. The pixel control asks whether the same deterministic action is recoverable in a full-resolution image-space formulation.

The development-only pixel smoke achieved 93.75% four-way retrieval and 98.4% true-vs-width pairwise accuracy, with 100% exact-oracle retrieval and all sanity checks passing. This is promising but not decisive because it used 64 test examples and one model seed.

Run the unchanged paired configuration only once using [`docs/pixel-control-paired-command.md`](docs/pixel-control-paired-command.md). The protocol is in [`docs/pixel-space-control-protocol.md`](docs/pixel-space-control-protocol.md).

## Key latent results

- [`docs/gate-2-results.md`](docs/gate-2-results.md)
- [`docs/gate-2-formal-retrieval-diagnostics.md`](docs/gate-2-formal-retrieval-diagnostics.md)
- [`results/gate2-formal/2026-08-21/`](results/gate2-formal/2026-08-21/)

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Do not rerun or retune the completed latent formal experiment. Do not alter the frozen paired pixel-control settings after seeing its result. Gate 3 planning remains blocked.
