# Latent Stroke Dynamics

Bachelor's thesis feasibility study on **action-conditioned canvas dynamics for stroke-based rendering**.

## Final experimental status

- **Gate 1 passed:** frozen DINOv2-small spatial features preserve one-stroke changes.
- **Latent Gate 2 failed:** strong average prediction but only 27.7% exact-action retrieval, dominated by width confusion.
- **Paired pixel-space control succeeded:** 100% exact-action retrieval across all three seeds.

The validation-selected pixel MLP has only 833 parameters. It reduced action-region pixel MSE by 99.950% versus identity and 99.948% versus mean delta, retained effectively complete performance at every crowding level and stress slice, and perfectly distinguished position, width, and intensity alternatives.

## Main conclusion

> Exact one-stroke dynamics are learnable by a tiny deterministic model in a full-resolution pixel formulation, but the tested frozen DINOv2 patch-token formulation does not retain enough predictive precision for exact action ranking despite low average latent error.

This localizes the failure to the **overall tested latent patch formulation**, not necessarily DINOv2 alone. The pixel control also changes target space, spatial resolution, and action-mask resolution.

See:

- [`docs/gate-2-results.md`](docs/gate-2-results.md)
- [`docs/gate-2-formal-retrieval-diagnostics.md`](docs/gate-2-formal-retrieval-diagnostics.md)
- [`docs/pixel-control-results.md`](docs/pixel-control-results.md)
- [`results/gate2-formal/2026-08-21/`](results/gate2-formal/2026-08-21/)
- [`results/pixel-control/2026-08-21/`](results/pixel-control/2026-08-21/)

## Next work

The core experiments are complete. Next work is figure review, paired latent-versus-pixel result writing, limitations, and thesis preparation—not additional tuning or Gate 3 planning.

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Do not rerun or retune the completed latent or pixel paired experiments. Preserve both the positive pixel result and negative latent retrieval result.
