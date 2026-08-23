# Guarded controlled latent-planner comparison

**Status:** Exactly one execution authorized  
**Smoke:** Completed and closed  
**Training/fine-tuning:** Not authorized

## Frozen comparison

- six untouched target seeds: `20261211`–`20261216`;
- six paired planner seeds: `20261221`–`20261226`;
- five methods: random, exact pixel, learned pixel, latent MSE, latent ranking;
- 100 executed strokes per target;
- 128 candidates per step;
- exact execution and observed-canvas re-encoding;
- fixed three-seed latent ensembles;
- no training or fine-tuning.

## Authorization evidence

The expanded suite passed 89 tests. Validation returned `latent_planner_controlled_runner_valid_unauthorized` while loading no model, generating no controlled target or planner data, and creating no output. Both atomic output paths were available and all criteria were already frozen.

## Single authorized command

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/17_latent_planner_controlled.py --controlled-run
```

Exactly one execution is authorized. It may take several minutes. Keep the machine awake and do not interrupt it. Do not rerun after either a completed output or a preserved `.incomplete` output appears.
