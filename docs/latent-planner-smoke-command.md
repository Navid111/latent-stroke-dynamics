# Guarded latent-planner implementation smoke

**Status:** Implemented; unauthorized validation pending  
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

## Validation-only command

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/16_latent_planner_smoke.py --validate-only
```

Validation must report status `latent_planner_smoke_runner_valid_unauthorized` with authorization false, models unloaded, target generation false, planner-data generation false, and both smoke output paths available.

## Guard boundary

Do not run `--smoke-run` while the config remains unauthorized. The command rejects that path before loading models, generating target seed `20261201`, or creating an output directory.

After validation is reviewed, one separate commit may set the smoke authorization. The smoke may then run once. Existing completed or `.incomplete` output directories are never overwritten.
