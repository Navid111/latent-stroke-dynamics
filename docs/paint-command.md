# User-facing image-to-strokes command

**Status:** Core suite passed; polarity repair awaiting local validation  
**Role:** Qualitative artifact only  
**Controlled Stage 3 result:** Frozen and unchanged

## Validation history

The initial packaged command passed the complete 36-test repository suite on the M1. The first arbitrary-image smoke then exposed a real preprocessing limitation: MNIST supplied a white digit on a black background, while the renderer starts white and can only add dark strokes. The planner therefore optimized the dominant black background instead of the digit.

That failed smoke is preserved as a qualitative failure case. It does not invalidate the synthetic controlled result because those targets already used the renderer-compatible dark-on-white convention.

The command now supports target polarity modes:

- `auto` — default; invert when the outer border is predominantly dark;
- `preserve` — never invert;
- `invert` — always invert.

Both the pre-polarity and normalized targets are saved, and the decision is recorded in the summary and run configuration.

## Current validation command

```bash
git pull --ff-only
source .venv/bin/activate
pytest
```

Two parameterized polarity checks were added. Expected total: `38 passed`.

## Repaired MNIST smoke

Only after all 38 tests pass:

```bash
python paint.py \
  --target "/Users/mohammednavid/Pictures/Example-of-a-MNIST-input-An-image-is-passed-to-the-network-as-a-matrix-of-28-by-28.webp" \
  --method learned \
  --polarity auto \
  --strokes 20 \
  --candidates 32 \
  --seed 20261001 \
  --output-dir outputs/qualitative-smoke-mnist-polarity-fixed
```

Do not delete or overwrite the original failed smoke. Compare its metrics and images with this repaired run.

## Full qualitative demonstration

Only after the repaired smoke artifacts are reviewed:

```bash
python paint.py \
  --target "/absolute/path/to/image.jpg" \
  --method learned \
  --polarity auto \
  --strokes 100 \
  --candidates 128 \
  --seed 20261001 \
  --output-dir outputs/qualitative-demo-1
```

`--method exact` and `--method random` are available for explicitly requested qualitative comparisons. Arbitrary-image outcomes are not a second controlled gate and cannot alter the frozen six-target result.

## Saved artifacts

Every completed output contains:

```text
processed_target_before_polarity.png
processed_target.png
initial_canvas.png
final_painting.png
progress.csv
summary.csv
summary.json
strokes.json
run_config.json
painting.gif
progress.png
comparison.png
frames/
```

The comparison image uses the normalized target and a fixed 0–255 absolute-error scale. Frame zero is the white initial canvas, followed by one frame per executed stroke.

## Safety and reproducibility guards

- Center-crop, EXIF correction, grayscale conversion, polarity handling, and 64×64 resizing are deterministic.
- Candidate generation and selection use a recorded seed.
- Learned selections are followed by exact stroke execution and replanning.
- The frozen checkpoint digest must match.
- Existing completed or `.incomplete` output directories are never overwritten.
- Artifacts are first written to `.incomplete` and atomically published only on success.
- Natural-image results remain qualitative and do not trigger checkpoint selection or retraining.
