# Current State

**Last updated:** 2026-08-18  
**Branch:** `main`  
**Current gate:** Gate 1 — frozen-encoder stroke sensitivity  
**Gate status:** Not yet run; no gate has passed

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

## Empirical status

No local installation, test run, model download, or experiment result has been reported yet. There is currently **no empirical evidence** that DINOv2 or any other frozen encoder passes Gate 1 for this setup.

## Current decisions

- Begin with 64×64 grayscale canvases.
- Begin with one straight-line stroke primitive.
- Start with `facebook/dinov2-small` as a convenient engineering baseline.
- Keep the encoder frozen.
- Compare global and spatial patch features.
- Test blank, moderately occupied, and crowded canvases.
- Do not train a dynamics predictor until Gate 1 has been evaluated.
- Start Gate 2 with a deterministic one-step predictor if Gate 1 passes.
- Treat depth-2 or depth-3 planning as optional.

## Next actions

1. Clone the repository locally.
2. Create and activate a Python virtual environment.
3. Install the project with `pip install -e ".[dev]"`.
4. Run `pytest` and fix only genuine setup or implementation failures.
5. Run the three-sample Gate 1 smoke test.
6. Inspect `distance_distributions.png`, `example_patch_heatmap.png`, and the CSV files.
7. If the smoke test is valid, run 25 samples at crowding levels 0, 5, and 15.
8. Record the exact command, hardware, runtime, errors, and observations in a dated logbook entry.
9. Update this file with the measured result and a justified Gate 1 decision.

## Commands

```bash
git clone https://github.com/Navid111/latent-stroke-dynamics.git
cd latent-stroke-dynamics
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell
pip install -e ".[dev]"
pytest
```

```bash
python experiments/01_embedding_sensitivity.py \
  --samples 3 \
  --crowding 0 5 \
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

- The environment and dependency installation have not been tested on the local machine.
- The first encoder run requires downloading pretrained weights.
- CPU execution may be slow.
- A non-zero representation distance is not sufficient evidence of usefulness.
- Heatmap localization is currently qualitative; a quantitative localization metric should be added only after the initial pipeline works.

## Handoff note

The next agent should help run or debug Gate 1. It should not begin the predictor, planner, reinforcement learning, complex brushes, or multi-step rollout unless this file is updated with evidence that the earlier gate passed.
