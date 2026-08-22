# Stage 3 random/exact smoke command — 2026-08-22

## Preconditions

The Stage 3 foundation passed all 30 tests locally on the base M1 MacBook Air:

```text
30 passed in 2.57s
```

## Command

```bash
git pull
source .venv/bin/activate
python experiments/04_pixel_planner_smoke.py
```

Defaults:

- one 64×64 synthetic target containing 20 strokes;
- target seed `20260901`;
- planner seed `20260822`;
- 20 executed strokes;
- 32 candidates per step;
- random and exact-greedy methods;
- deterministic exact replay check;
- development-only output under `outputs/stage3-smoke-1/`.

## Expected artifacts

```text
outputs/stage3-smoke-1/
├── target.png
├── summary.csv
├── run_config.json
├── progress_curves.png
├── final_comparison.png
├── random/
│   ├── initial_canvas.png
│   ├── final_canvas.png
│   ├── progress.csv
│   ├── strokes.json
│   └── painting.gif
└── exact/
    ├── initial_canvas.png
    ├── final_canvas.png
    ├── progress.csv
    ├── strokes.json
    └── painting.gif
```

## What to send back

1. Complete terminal output.
2. `summary.csv`.
3. `run_config.json`.
4. `final_comparison.png`.
5. `progress_curves.png`.

The GIFs can remain local unless a visual issue appears in the static outputs.

## Review rule

This smoke checks deterministic execution, output serialization, whether exact greedy behavior is sensibly stronger than random, and whether the 20-step output looks structurally plausible. It cannot change any frozen earlier result and does not make the six-target controlled decision.
