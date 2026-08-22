# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Validate best-painting output, then freeze the representation extension  
**Status:** MNIST learned-versus-exact qualitative diagnostic complete

## Frozen experimental chain

- Gate 1 passed.
- Latent DINOv2 Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval.
- Controlled Stage 3 learned planning succeeded across six synthetic targets.

Do not rerun, retune, relabel, or replace these results.

## Controlled Stage 3

- learned mean final MSE: `0.060032`;
- random mean final MSE: `0.155971`;
- exact mean final MSE: `0.050479`;
- learned reduction versus random: `61.51%`;
- learned/exact ratio: `1.18925`;
- integrity and deterministic replay passed.

## Qualitative MNIST result

Auto polarity normalization repaired the initial light-on-dark mismatch. At 100 strokes and 128 candidates:

| Metric | Learned | Exact |
|---|---:|---:|
| Final MSE | `0.040860` | `0.022196` |
| Improvement | `71.70%` | `84.62%` |
| Improving steps | `42` | `88` |
| Best step | `33` | `99` |

Learned top-1/top-5 exact agreement was 23%/37%, mean exact rank was 9.23 of 128, and mean regret was `0.000410`. Exact continued improving almost to the budget while learned deteriorated after step 33. This localizes the main late-stage problem to learned ranking under long, crowded qualitative trajectories rather than candidate availability alone.

The exact method is currently the strongest arbitrary-image artifact. The learned method remains a controlled success with a documented natural-target/long-horizon limitation.

## Painter update

The command now saves `best_painting.png` separately from the requested-budget `final_painting.png`. Summary metadata records best step, best MSE/MAE, best improvement, and final-versus-best degradation. The comparison figure shows both states.

## Immediate next action

```bash
git pull --ff-only
source .venv/bin/activate
pytest
```

Expected: `38 passed`. No MNIST rerun is required; its step-33 canvas already exists at `outputs/qualitative-demo-mnist-3-learned/frames/frame_0033.png`.

After validation, freeze the post-core representation-extension protocol before generating new data.

## Boundaries

- Preserve all failed and successful qualitative outputs.
- Do not use qualitative images to retrain the pixel checkpoint.
- No RL, color, textured brushes, or multi-step rollout before thesis completion.
