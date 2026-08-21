# Latent Stroke Dynamics

Bachelor's thesis project on action-conditioned canvas dynamics and sequential stroke-based rendering.

## Completed foundation

- **Gate 1 passed:** frozen DINOv2-small spatial features preserve one-stroke changes.
- **Latent Gate 2 failed:** strong average prediction but only 27.7% exact-action retrieval, dominated by width confusion.
- **Paired pixel control succeeded:** 100% exact-action retrieval across all three seeds.

These results motivate a scoped pixel-space pivot rather than abandoning the final painting artifact.

## Active Stage 3

Build a command-line painter that accepts an image and constructs a 64×64 grayscale approximation line by line.

Required methods:

1. random candidate selection;
2. exact-renderer greedy pixel planning;
3. learned pixel-predictor planning with exact stroke execution.

The controlled comparison uses six synthetic targets, 100 strokes, and 128 candidates per step. Arbitrary user images are qualitative demonstrations.

See:

- [`docs/stage-3-pixel-planner-protocol.md`](docs/stage-3-pixel-planner-protocol.md)
- [`docs/final-artifact-roadmap.md`](docs/final-artifact-roadmap.md)
- [`docs/latent-vs-pixel-comparison.md`](docs/latent-vs-pixel-comparison.md)

## Main scientific conclusion so far

> Exact one-stroke dynamics are learnable by a tiny deterministic model in a full-resolution pixel formulation, while the tested frozen DINOv2 patch-token formulation does not preserve enough predictive precision for exact action ranking.

Stage 3 now tests whether the successful pixel predictor can rank candidate strokes toward a target and produce a working sequential painting artifact.

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Do not rerun or retune the completed paired experiments. Keep the final artifact grayscale, 64×64, straight-line, and one-step greedy until the controlled comparison and thesis are complete.
