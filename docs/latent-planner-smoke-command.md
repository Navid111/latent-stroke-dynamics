# Guarded latent-planner implementation smoke

**Status:** One execution authorized after successful no-data validation  
**Role:** one-target engineering diagnostic only  
**Controlled comparison:** unauthorized

## Frozen smoke

- methods: random, exact pixel, learned pixel, latent MSE ensemble, latent ranking ensemble;
- target seed: `20261201`;
- planner seed: `20261202`;
- target: 20 synthetic strokes on a 64×64 grayscale canvas;
- budget: 20 executed strokes and 32 candidates per step;
- latent models: fixed three-seed means using seeds 11, 22, and 33;
- exact stroke execution and exact-canvas re-encoding after every latent selection;
- no model training or fine-tuning.

## Validation evidence

The full suite passed all 84 tests. Validation returned `latent_planner_smoke_runner_valid_unauthorized` with models unloaded, target generation false, planner-data generation false, both smoke paths available, and controlled authorization false.

## Single authorized command

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/16_latent_planner_smoke.py --smoke-run
```

Exactly one execution is authorized. Do not rerun it after either a completed output or a preserved `.incomplete` output appears. The runner enforces both conditions.

The smoke is implementation-diagnostic only. It cannot alter models, choose settings, revise formal results, or authorize the controlled comparison.
