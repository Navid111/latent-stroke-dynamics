# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Validate the user-facing painter, then freeze the representation extension  
**Status:** Painter implementation committed; local test validation pending

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

The completed controlled output must be preserved and the comparison must not be run again.

## User-facing painter

Committed files:

- `paint.py`;
- `src/latent_stroke_dynamics/painting_cli.py`;
- `tests/test_painting_cli.py`;
- `docs/paint-command.md`.

The command supports random, exact, and learned planning and saves the processed target, final painting, metrics, ordered strokes, frames, GIF, progress plot, and fixed-scale comparison. It verifies the frozen checkpoint, refuses output overwrite, preserves incomplete work, and labels all user images qualitative.

## Immediate next action

```bash
git pull --ff-only
source .venv/bin/activate
pytest
```

Expected total after the three new painter tests: `36 passed`.

Do not run a user image until the full suite passes. After validation, use a 20-stroke/32-candidate learned smoke before a full 100-stroke/128-candidate demonstration.

## Authorized later extension

After the painter is secure, freeze a new post-core protocol comparing:

1. existing DINOv2 latent result;
2. one reconstruction-oriented frozen encoder;
3. one small task-trained spatial latent encoder;
4. existing raw-pixel control.

This must use new untouched data seeds and cannot modify earlier gates.

## Boundaries

- Grayscale 64×64 straight-line output.
- Exact execution after learned selection.
- No RL, color, textured brushes, or multi-step rollout before thesis completion.
- Natural-image results are qualitative, not a second formal gate.
