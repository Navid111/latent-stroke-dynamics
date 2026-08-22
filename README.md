# Latent Stroke Dynamics

Bachelor's thesis project on action-conditioned canvas dynamics and sequential stroke-based rendering.

## Completed controlled findings

- **Gate 1 passed:** frozen DINOv2-small spatial features preserve one-stroke changes.
- **Latent Gate 2 failed:** strong average prediction but only 27.7% exact-action retrieval, dominated by width confusion.
- **Paired pixel control succeeded:** 100% exact-action retrieval across all three seeds.
- **Stage 3 controlled painter succeeded:** across six fixed targets, learned planning reduced mean final MSE by 61.51% versus random and finished 18.93% above exact greedy while improving all six targets.

These results establish the experimental core while preserving the distinction between the failed latent formulation and the successful pixel-space pivot.

## Current stage

Package the frozen learned planner as a command-line painter that accepts an arbitrary image and constructs a 64×64 grayscale approximation line by line. Then run a small qualitative demonstration set and integrate the controlled evidence into the thesis.

Stage 3 methods:

1. random candidate selection;
2. exact-renderer greedy pixel planning;
3. learned pixel-predictor planning with exact stroke execution.

The completed controlled comparison used six synthetic targets, 100 strokes, and 128 candidates per step. Arbitrary user images remain qualitative demonstrations.

See:

- [`docs/stage-3-controlled-results.md`](docs/stage-3-controlled-results.md)
- [`docs/stage-3-pixel-planner-protocol.md`](docs/stage-3-pixel-planner-protocol.md)
- [`docs/final-artifact-roadmap.md`](docs/final-artifact-roadmap.md)
- [`docs/latent-vs-pixel-comparison.md`](docs/latent-vs-pixel-comparison.md)

## Main scientific conclusion

> Frozen DINOv2 patch-token changes were sensitive to individual strokes but did not preserve enough predictive precision for exact latent action ranking. A tiny full-resolution pixel dynamics model recovered one-step action precision and supported successful sequential target-guided painting: under a frozen six-target protocol, learned planning substantially outperformed random selection and approached an exact one-step greedy renderer oracle.

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Do not rerun or retune the completed controlled experiments. Keep the final artifact grayscale, 64×64, straight-line, and one-step greedy until the qualitative demonstrations and thesis are complete.
