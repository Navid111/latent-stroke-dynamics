# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Stage 3 pixel-space target-guided painter  
**Status:** All-method smoke passed; controlled config frozen and awaiting validation

## Frozen completed foundation

- Gate 1 passed.
- Latent Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval across all three seeds.
- Demonstration checkpoint trained from train/validation data only.

Do not rerun, retune, relabel, or replace these results.

## Stage 3 engineering evidence

- repository suite: `33 passed in 2.12s`;
- random/exact smoke: pass;
- all-method smoke: pass;
- checkpoint digest: `e32f3612f7a184e4e9b58f95a987551bd25cdb17ff1bf2b6be40fcf5781ea472`;
- deterministic learned replay: pass.

All-method smoke final MSE:

- random: `0.144913`;
- exact: `0.074427`;
- learned: `0.089196`.

Learned improved 40.32% from white, finished 38.45% below random and 19.84% above exact, and improved 19 of 20 actions. Exact top-1/top-5 agreement was 55%/80%, with mean exact rank 4.6 and small mean regret.

## Controlled Stage 3

Frozen config:

- six targets with seeds `20260901`–`20260906`;
- planner seeds `20260920`–`20260925`;
- 100 selected strokes;
- 128 candidates per step;
- methods random, exact, and learned;
- fixed checkpoint and success criteria.

Files:

- `configs/stage3-controlled-2026-08-22.json`;
- `experiments/07_pixel_planner_controlled.py`;
- `docs/stage-3-controlled-command.md`.

## Next action

Run validation only:

```bash
git pull
source .venv/bin/activate
python experiments/07_pixel_planner_controlled.py --validate-only
```

Do not start the controlled run if validation fails or if a completed/incomplete controlled output directory already exists.

## Boundaries

- No controlled outputs currently exist.
- Do not change the frozen config or thresholds.
- Run the controlled comparison only once after validation.
- Preserve valid pass or fail results without retuning.
- The learned planner always executes chosen strokes with the exact renderer.
