# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Repair and revalidate qualitative target preprocessing  
**Status:** 36-test painter suite passed; MNIST smoke exposed polarity mismatch; repair committed

## Completed experimental chain

- Gate 1 passed: frozen DINOv2 spatial features preserve localized stroke changes.
- Latent Gate 2 formally failed: exact-action retrieval was 27.7% despite strong average-error improvement.
- Paired pixel control succeeded: 100% retrieval across all three seeds.
- Stage 3 controlled planner succeeded on the single authorized six-target run.

Do not rerun, retune, relabel, or replace these results.

## Stage 3 controlled decision

- learned improved all six targets;
- learned mean final MSE: `0.060032`;
- random mean final MSE: `0.155971`;
- exact mean final MSE: `0.050479`;
- learned reduction versus random: `61.51%`;
- learned/exact ratio: `1.18925`;
- implementation integrity and deterministic replay passed.

The completed controlled output must be preserved and the comparison must not be run again.

## User-facing painter validation

The first implementation passed `36 tests in 2.81s`, including all artifact and overwrite guards.

The first learned arbitrary-image smoke used an MNIST-style white digit on a black background. Its result was poor:

- initial MSE `0.798784`;
- final MSE `0.533353`;
- 33.23% reduction;
- top-1/top-5 exact agreement `15%` / `45%`;
- mean exact rank `11.75` of 32;
- mean regret `0.011429`.

This is a renderer/target polarity mismatch. The painter starts with white and draws only values `{0, 32, 64, 96, 128}`. It cannot paint a white digit onto a black canvas, so pixel MSE drives strokes across the black background. The original failed output is a valid qualitative limitation and must be preserved.

## Repair

The qualitative command now defaults to `--polarity auto`. It inspects the resized target border and inverts targets whose border median is below `127.5`, mapping light-on-dark inputs to the trained dark-on-white convention. `preserve` and `invert` overrides are available. Both pre-normalization and normalized targets are saved; metadata records the decision.

This changes only arbitrary-image preprocessing. It does not change the renderer, learned checkpoint, proposal logic, controlled targets, controlled metrics, or controlled decision.

## Immediate next action

```bash
git pull --ff-only
source .venv/bin/activate
pytest
```

Expected total after two parameterized polarity tests: `38 passed`.

If validation passes, rerun the same 20-stroke/32-candidate MNIST smoke in a new output directory with `--polarity auto`. Do not run the 100-stroke version yet.

## Authorized later extension

After the painter is secure, freeze a new post-core protocol comparing existing DINOv2, one reconstruction-oriented frozen encoder, one task-trained spatial latent encoder, and the existing raw-pixel control on untouched seeds.

## Boundaries

- Grayscale 64×64 straight-line output.
- Exact execution after learned selection.
- No RL, color, textured brushes, or multi-step rollout before thesis completion.
- Natural-image results are qualitative, not a second formal gate.
