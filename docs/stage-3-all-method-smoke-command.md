# Stage 3 all-method smoke command — 2026-08-22

## Preconditions

- `33 passed in 2.12s`;
- checkpoint training status `success`;
- selected checkpoint epoch 29;
- validation balanced MSE `0.000164624`;
- checkpoint reload identity and state-dict integrity passed.

## Command

```bash
git pull
source .venv/bin/activate
python experiments/06_pixel_planner_all_methods_smoke.py
```

The script uses the existing local checkpoint automatically:

```text
checkpoints/stage3-pixel-mlp-seed11.pt
```

Defaults remain development-only:

- one fixed 20-stroke synthetic target;
- 20 selected strokes;
- 32 candidates per step;
- random, exact greedy, and learned pixel planning;
- exact execution for every selected action;
- deterministic learned replay;
- learned exact rank, top-1, top-5, and regret diagnostics.

## What to send back

1. Complete terminal output.
2. `outputs/stage3-all-methods-smoke-1/summary.csv`.
3. `outputs/stage3-all-methods-smoke-1/run_config.json`.
4. `outputs/stage3-all-methods-smoke-1/learned_step_diagnostics.csv`.
5. `outputs/stage3-all-methods-smoke-1/final_comparison.png`.
6. `outputs/stage3-all-methods-smoke-1/progress_curves.png`.

The five files can be uploaded together; paste the terminal output separately. GIFs can remain local unless the static figures reveal an issue.

## Review purpose

This smoke checks whether the saved learned predictor can rank target-directed candidate strokes, how closely it follows exact greedy selection, whether its exact regret remains small, and whether output artifacts work for all three methods. It cannot make the six-target formal Stage 3 decision.
