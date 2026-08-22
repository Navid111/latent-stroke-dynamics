# User-facing image-to-strokes command

**Status:** Polarity repair validated; best-painting artifact awaiting local validation  
**Role:** Qualitative artifact only  
**Controlled Stage 3 result:** Frozen and unchanged

## Validation history

The packaged painter passed 36 tests. The first arbitrary-image smoke then exposed a target-polarity mismatch: MNIST supplied a white digit on black while the renderer starts white and adds dark strokes. Auto polarity normalization repaired the mismatch, and the expanded suite passed all 38 tests.

The full MNIST diagnostic showed that learned error reached its minimum at step 33 and then rose, whereas exact greedy continued improving until step 99. The command therefore now preserves both:

- `best_painting.png` — lowest true target MSE seen anywhere in the requested trajectory;
- `final_painting.png` — literal canvas after the requested number of strokes.

The summary records best step/error and final-versus-best degradation. The comparison figure shows target, best, final, and fixed-scale errors. This is output selection only; the planner trajectory is unchanged.

## Current validation command

```bash
git pull --ff-only
source .venv/bin/activate
pytest
```

Expected total remains `38 passed`; existing artifact tests now also verify best-painting output and metrics.

## Standard qualitative command

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

For the strongest current arbitrary-image output, use `--method exact`. Use `--method learned` when demonstrating the learned dynamics result and its deployment limitations. User images are qualitative and cannot alter the frozen controlled result or checkpoint.

## Polarity modes

- `auto` — default; invert when the outer border is predominantly dark;
- `preserve` — never invert;
- `invert` — always invert.

Both the pre-polarity and normalized targets are saved, and the decision is recorded in the summary and configuration.

## Saved artifacts

```text
processed_target_before_polarity.png
processed_target.png
initial_canvas.png
best_painting.png
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

## Safety and reproducibility guards

- Preprocessing, polarity handling, candidate generation, and selection are deterministic.
- Learned selections are followed by exact execution and replanning.
- The frozen checkpoint digest must match.
- Existing completed or `.incomplete` output directories are never overwritten.
- Artifacts are atomically published only on success.
- Natural-image outcomes do not trigger checkpoint selection or retraining.
