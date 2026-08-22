# Latent Stroke Dynamics

Bachelor's thesis project on action-conditioned canvas dynamics and sequential stroke-based rendering.

## Completed controlled findings

- **Gate 1 passed:** frozen DINOv2-small spatial features preserve one-stroke changes.
- **Latent Gate 2 failed:** strong average prediction but only 27.7% exact-action retrieval, dominated by width confusion.
- **Paired pixel control succeeded:** 100% exact-action retrieval across all three seeds.
- **Stage 3 controlled painter succeeded:** across six fixed targets, learned planning reduced mean final MSE by 61.51% versus random and finished 18.93% above exact greedy while improving all six targets.

These results establish the experimental core while preserving the distinction between the failed latent formulation and the successful pixel-space pivot.

## User-facing painter

After installing the project and restoring the validated local checkpoint, paint an arbitrary image with:

```bash
python paint.py \
  --target path/to/image.jpg \
  --method learned \
  --strokes 100 \
  --candidates 128 \
  --output-dir outputs/my-painting
```

The command center-crops the target to 64×64 grayscale and saves the final painting, ordered stroke JSON, metrics, progress figure, individual frames, comparison figure, and GIF. User images are qualitative demonstrations and do not alter the controlled result.

See [`docs/paint-command.md`](docs/paint-command.md) for validation and smoke instructions.

## Current stage

1. Validate the user-facing painter locally.
2. Run a small fixed qualitative demonstration set.
3. Freeze a compact post-core representation extension comparing the existing DINOv2 result with one reconstruction-oriented frozen encoder and one task-trained latent encoder.
4. Integrate the complete evidence into the thesis.

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

Do not rerun or retune the completed controlled experiments. Keep the final artifact grayscale, 64×64, straight-line, and one-step greedy. Any representation extension must receive a new protocol and must not rewrite the existing formal decisions.
