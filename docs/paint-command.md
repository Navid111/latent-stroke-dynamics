# User-facing image-to-strokes command

**Status:** Implementation committed; local test validation pending  
**Role:** Qualitative artifact only  
**Controlled Stage 3 result:** Frozen and unchanged

## First validation

Pull the implementation and run the complete repository suite before painting a user image:

```bash
git pull --ff-only
source .venv/bin/activate
pytest
```

The previous suite contained 33 tests. The painter adds three tests, so the expected total is 36 passing tests.

## Qualitative smoke

After the tests pass, choose one local image with a clear subject and good contrast. Start with a reduced-budget smoke:

```bash
python paint.py \
  --target "/absolute/path/to/image.jpg" \
  --method learned \
  --strokes 20 \
  --candidates 32 \
  --seed 20261001 \
  --output-dir outputs/qualitative-smoke-1
```

The learned command verifies the exact frozen checkpoint metadata and SHA-256 before painting. It never retrains the model.

## Full qualitative demonstration

Only after the smoke artifacts are reviewed:

```bash
python paint.py \
  --target "/absolute/path/to/image.jpg" \
  --method learned \
  --strokes 100 \
  --candidates 128 \
  --seed 20261001 \
  --output-dir outputs/qualitative-demo-1
```

`--method exact` and `--method random` are available for explicitly requested qualitative comparisons. Arbitrary-image outcomes are not a second controlled gate and cannot alter the frozen six-target result.

## Saved artifacts

Every completed output contains:

```text
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

The comparison image uses a fixed 0–255 absolute-error scale. Frame zero is the white initial canvas, followed by one frame per executed stroke.

## Safety and reproducibility guards

- Center-crop, EXIF correction, grayscale conversion, and 64×64 resizing are deterministic.
- Candidate generation and selection use a recorded seed.
- Learned selections are followed by exact stroke execution and replanning.
- The frozen checkpoint digest must match.
- Existing completed or `.incomplete` output directories are never overwritten.
- Artifacts are first written to `.incomplete` and atomically published only on success.
- Natural-image results remain qualitative and do not trigger checkpoint selection or retraining.
