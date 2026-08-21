# Latent Stroke Dynamics

Bachelor's thesis experiments on **action-conditioned latent canvas dynamics for stroke-based rendering**.

## Current status

- **Gate 1 passed** on 2026-08-19.
- **Gate 2 formally failed** on 2026-08-21 under its frozen conjunctive rule.
- **Pixel-space explanatory control protocol frozen** on 2026-08-21 before implementation.

The latent model predicted broad one-step consequences strongly but retrieved the exact outcome only 27.7%. It was reliable for position and intensity yet systematically confused stroke width.

## Active experiment

The next experiment is a minimal action-conditioned pixel-space predictor using the same deterministic transition distribution. It predicts normalized 64×64 pixel residuals from:

- the current pixel value;
- the same seven-value stroke vector;
- an exact full-resolution proposed-stroke mask;
- normalized pixel coordinates.

The comparison retains identity, mean-delta, linear, small nonlinear, and exact compositing-oracle methods. See [`docs/pixel-space-control-protocol.md`](docs/pixel-space-control-protocol.md).

The control asks whether exact action information is recoverable in pixel space. It cannot revise the completed latent Gate 2 decision, and Gate 3 planning remains blocked.

## Key results

- [`docs/gate-2-results.md`](docs/gate-2-results.md)
- [`docs/gate-2-formal-visual-review.md`](docs/gate-2-formal-visual-review.md)
- [`docs/gate-2-formal-retrieval-diagnostics.md`](docs/gate-2-formal-retrieval-diagnostics.md)
- [`results/gate2-formal/2026-08-21/`](results/gate2-formal/2026-08-21/)

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Do not rerun or retune the completed latent formal experiment. Any contrastive or width-specific latent objective is future work, not a replacement for the recorded result.
