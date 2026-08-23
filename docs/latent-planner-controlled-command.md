# Guarded controlled latent-planner comparison

**Status:** Implemented; validation-only required  
**Controlled authorization:** False  
**Permitted next action:** no-data validation only

## Frozen comparison

- six untouched target seeds: `20261211`–`20261216`;
- six paired planner seeds: `20261221`–`20261226`;
- five methods: random, exact pixel, learned pixel, latent MSE, latent ranking;
- 100 executed strokes per target;
- 128 candidates per step;
- exact execution and observed-canvas re-encoding;
- fixed three-seed latent ensembles;
- no training or fine-tuning.

## Validation command

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/17_latent_planner_controlled.py --validate-only
```

Validation must not load models, generate any controlled target, create either output directory, or authorize the comparison. Expected status: `latent_planner_controlled_runner_valid_unauthorized`.

Do not run `--controlled-run`. A separate authorization commit is required after successful validation.
