# Current State

**Last updated:** 2026-08-19  
**Branch:** `main`  
**Current gate:** Gate 1 — frozen-encoder stroke sensitivity  
**Gate status:** Encoding works; plotting compatibility fix committed; no gate has passed

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
- Fixed the omitted `torchvision` dependency.
- Successfully downloaded and loaded `facebook/dinov2-small`.
- Successfully encoded all smoke-test canvases on CPU in the second attempt.
- Diagnosed a Matplotlib boxplot API incompatibility and committed a compatibility fix.
- Recorded both setup attempts in `logbooks/2026-08-19.md`.

## Empirical status

The second smoke test completed representation extraction but stopped while producing the distribution plot because the installed Matplotlib version rejected the old `labels` argument. The plotting fix is now on `main` but has not yet been rerun locally.

CSV files may have been partially written, but the smoke test is not considered complete until the script reaches its final message and produces all expected artifacts. There is still no justified Gate 1 pass or fail decision.

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

1. Pull the Matplotlib compatibility fix from `main`.
2. Re-run the same three-sample Gate 1 smoke command.
3. Confirm that the script reaches its final “Saved Gate 1 results” message.
4. Inspect and share the printed mean-distance table.
5. Inspect `distance_distributions.png` and `example_patch_heatmap.png`.
6. If the smoke test is structurally valid, run 25 samples at crowding levels 0, 5, and 15.
7. Record the full result and a justified Gate 1 decision.

## Immediate commands

```bash
git pull
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

- The plotting compatibility fix has not yet been verified locally.
- Python 3.14 is newer than many research stacks, although the current PyTorch, torchvision, Transformers, and DINOv2 inference path now work.
- A non-zero representation distance is not sufficient evidence of usefulness.
- Heatmap localization is currently qualitative; a quantitative localization metric should be added only after the initial pipeline works.

## Handoff note

The next agent should complete and inspect Gate 1. It should not begin the predictor, planner, reinforcement learning, complex brushes, or multi-step rollout unless this file is updated with evidence that the earlier gate passed.
