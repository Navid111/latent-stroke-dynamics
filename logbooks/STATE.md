# Current State

**Last updated:** 2026-08-19  
**Branch:** `main`  
**Current gate:** Gate 1 — frozen-encoder stroke sensitivity  
**Gate status:** Smoke test blocked by a dependency error; no gate has passed

## Objective

Determine whether frozen spatial visual features reliably preserve the local change caused by one controlled stroke. This evidence is required before training an action-conditioned next-representation predictor.

## Completed

- Created the private `latent-stroke-dynamics` repository.
- Added a Python project scaffold and installation metadata.
- Implemented a deterministic grayscale straight-line renderer.
- Added renderer unit tests.
- Added a frozen Hugging Face vision-encoder wrapper exposing global and patch features.
- Added the initial Gate 1 embedding-sensitivity experiment.
- Added result export, aggregate summaries, distribution plots, and an example patch heatmap.
- Added the Gate 1 protocol, thesis plan, and agent operating instructions.
- Created a local virtual environment on a base-model M1 MacBook Air.
- Attempted the first CPU smoke test.
- Diagnosed the first setup failure: `torchvision` was missing from project dependencies.
- Added `torchvision` to `pyproject.toml` and recorded the failure in `logbooks/2026-08-19.md`.

## Empirical status

The first smoke test stopped during `AutoImageProcessor` initialization because `torchvision` was not installed. No pretrained model was loaded and no canvases were encoded. This is a setup failure, **not** evidence for or against the research hypothesis.

There is still no empirical evidence that DINOv2 or any other frozen encoder passes Gate 1 for this setup.

## Current decisions

- Begin with 64×64 grayscale canvases.
- Begin with one straight-line stroke primitive.
- Start with `facebook/dinov2-small` as a convenient engineering baseline.
- Keep the encoder frozen.
- Compare global and spatial patch features.
- Test blank, moderately occupied, and crowded canvases.
- Use CPU with batch size 4 for the first M1 smoke test.
- Do not train a dynamics predictor until Gate 1 has been evaluated.
- Start Gate 2 with a deterministic one-step predictor if Gate 1 passes.
- Treat depth-2 or depth-3 planning as optional.

## Next actions

1. Pull the dependency fix from `main`.
2. Re-run `python -m pip install -e ".[dev]"` inside the active environment.
3. Verify that both `torch` and `torchvision` import successfully.
4. Run `pytest` and report its output if it has not already been run.
5. Re-run the three-sample Gate 1 smoke test on CPU with batch size 4.
6. Inspect `distance_distributions.png`, `example_patch_heatmap.png`, and the CSV files.
7. If the smoke test is valid, run 25 samples at crowding levels 0, 5, and 15.
8. Record the exact command, hardware, runtime, errors, and observations in the dated logbook.
9. Update this file with the measured result and a justified Gate 1 decision.

## Immediate commands

```bash
git pull
python -m pip install -e ".[dev]"
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
pytest
```

```bash
python experiments/01_embedding_sensitivity.py \
  --samples 3 \
  --crowding 0 5 \
  --batch-size 4 \
  --device cpu \
  --output-dir outputs/gate1-smoke
```

## Expected Gate 1 artifacts

- `results.csv`
- `aggregate_summary.csv`
- `distance_distributions.png`
- `example_patch_heatmap.png`
- `run_config.json`

These generated artifacts are ignored by Git. Preserve final evidence by summarizing it in a dated log and later copying selected thesis-ready figures into a tracked `figures/` directory.

## Current blockers and risks

- The missing `torchvision` dependency has been fixed in the repository but not yet verified locally.
- The active environment uses Python 3.14, which is newer than many research stacks. If compatible wheels or imports fail, use Python 3.12.
- The first encoder run still requires downloading pretrained weights.
- CPU execution may be slow.
- A non-zero representation distance is not sufficient evidence of usefulness.
- Heatmap localization is currently qualitative; a quantitative localization metric should be added only after the initial pipeline works.

## Handoff note

The next agent should finish debugging and run Gate 1. It should not begin the predictor, planner, reinforcement learning, complex brushes, or multi-step rollout unless this file is updated with evidence that the earlier gate passed.
