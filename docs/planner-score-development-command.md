# Planner-score long-horizon development validation

The candidate-score audit is archived and closed. Its frozen winner is the MSE-only three-seed ensemble scored by normalized-latent L1.

A guarded planner-development runner is implemented, but planner-development data are not authorized.

## Run now

```bash
git pull --ff-only
pytest
python experiments/19_planner_score_development.py --validate-only
```

Expected test count: **103 passed**.

Expected validation status:

```text
planner_score_planner_development_runner_valid_unauthorized
```

The JSON should report:

- the closed score-audit selection verified;
- three reserved new target seeds and three reserved planner seeds;
- five fixed methods;
- 100 maximum steps and 128 candidates per step;
- selected predictor `mse_only` and score `normalized_latent_l1`;
- no-op margin `0.0`;
- score audit, planner development, and confirmatory runs unauthorized;
- all frozen model/hash references verified;
- no models loaded;
- no targets or planner data generated;
- no training or fine-tuning.

## Do not run

```bash
python experiments/19_planner_score_development.py --planner-development
```

That command must remain blocked until the tests and validation output are reviewed and a separate single-run authorization is committed.
