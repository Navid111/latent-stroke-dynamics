# High-resolution qualitative stroke replay

## Purpose

The existing image-to-strokes painter plans on a 64x64 grayscale canvas. Merely
resizing its final raster makes the square pixels larger. The replay command
instead redraws the already-selected normalized stroke sequence at 256x256,
512x512, or another presentation resolution and scales stroke widths with the
canvas.

The replay uses two-times supersampling and Lanczos downsampling by default, so
stroke edges are smoother than nearest-neighbor enlargement. It does not add
new geometric information or improve candidate selection; it only produces a
cleaner rendering of the same decisions.

## Scientific boundary

This command:

- reads an existing completed qualitative `paint.py` output;
- preserves the source stroke order, coordinates, intensity, and best-step index;
- performs no candidate generation or selection;
- loads no model or checkpoint;
- performs no training;
- recomputes no evaluation metric;
- verifies the source artifacts are byte-identical after replay;
- writes to a separate atomically finalized output directory;
- does not change any controlled or formal result.

It is a presentation feature, not a new experiment.

## Legacy compatibility

Early qualitative output directories predate the addition of `best_step` to
`summary.json`. For those outputs, the replay deterministically reconstructs
the best step from the saved `progress.csv` curve. The replay configuration
records whether the step came from `summary.json` or `progress.csv`.

## Validation

From the repository root:

```bash
git pull --ff-only
source .venv/bin/activate
python -m pytest -q
```

Expected total after the legacy-compatibility repair: `168 passed`.

## Usage

First produce or identify a completed qualitative painting directory created by
`paint.py`. Then run, for example:

```bash
python replay_high_res.py \
  --painting-dir outputs/qualitative-demo-mnist-3-learned \
  --output-dir outputs/qualitative-demo-mnist-3-learned-512 \
  --size 512 \
  --supersample 2
```

If your completed painting has another path, replace `--painting-dir` with that
path. Never pass a `.incomplete` directory.

## Saved artifacts

```text
reference.png
initial.png
best.png
final.png
painting.gif
replay_config.json
```

`best.png` uses the best step recorded during the original 64x64 planning run,
or the exact minimum reconstructed from `progress.csv` for a legacy run. The
replay does not select a new best frame at presentation resolution.

## What this does not solve

- It does not make the underlying planner choose better strokes.
- It does not add color, opacity, curves, or texture.
- It does not turn the Phase B0 one-step prediction result into successful
  long-horizon planning.

RGB strokes and coarse-to-fine scheduling should be implemented as separate
qualitative extensions so their effects remain distinguishable.
