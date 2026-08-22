# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Package the successful Stage 3 painter and run qualitative demos  
**Status:** Controlled Stage 3 succeeded; result frozen and archived

## Completed experimental chain

- Gate 1 passed: frozen DINOv2 spatial features preserve localized stroke changes.
- Latent Gate 2 formally failed: exact-action retrieval was 27.7% despite strong average-error improvement.
- Paired pixel control succeeded: 100% retrieval across all three seeds.
- Stage 3 controlled planner succeeded on the single authorized six-target run.

Do not rerun, retune, relabel, or replace these results.

## Stage 3 controlled decision

- controlled eligible: true;
- control status: success;
- implementation integrity: passed;
- learned improved all six targets;
- learned mean final MSE: `0.060032`;
- random mean final MSE: `0.155971`;
- exact mean final MSE: `0.050479`;
- learned reduction versus random: `61.51%`;
- learned/exact ratio: `1.18925`;
- exact top-1/top-5 agreement: `33.5%` / `58.67%`;
- mean exact rank: `7.51` of 128;
- mean one-step regret: `0.0003867`;
- deterministic replay: passed.

All four frozen criteria passed. The completed output directory must be preserved and the controlled comparison must not be run again.

## Interpretation

The learned model need not select the one-step exact candidate every time. Its average regret remained small enough to build target-aligned paintings and stay close to exact greedy over 100 replanning steps. Target 5 finished better under learned planning than exact greedy because exact is only a one-step oracle and different trajectories induce different future proposal sets.

## Frozen artifacts

- local controlled outputs: `outputs/stage3-controlled-2026-08-22`;
- archived raw summaries: `results/stage3-controlled-2026-08-22`;
- result review: `docs/stage-3-controlled-results.md`;
- checkpoint digest: `e32f3612f7a184e4e9b58f95a987551bd25cdb17ff1bf2b6be40fcf5781ea472`.

## Next action

Build the user-facing arbitrary-image command around the frozen learned planner. It must save:

- processed target image;
- final painting;
- ordered stroke JSON;
- per-step metrics CSV;
- progress plot;
- frames and GIF;
- run configuration and checkpoint digest.

Then run a small, fixed qualitative set containing both easy and difficult images. Do not use qualitative outcomes to alter the controlled result or retrain the checkpoint.

## Boundaries

- Grayscale 64×64 straight-line output.
- Exact execution after learned selection.
- No RL, color, textured brushes, or multi-step rollout before thesis completion.
- Natural-image results are qualitative, not a second formal gate.
