# Planner-score audit validation command

The Stage A score-audit protocol is frozen, but development data are not authorized.

## Run now

```bash
git pull --ff-only
pytest
python experiments/18_planner_score_alignment.py --validate-only
```

Expected test count: **96 passed**.

Expected validation status:

```text
planner_score_audit_runner_valid_unauthorized
```

The JSON should also report:

- 8 reserved development targets;
- 9 fixed states per target;
- 72 candidate sets;
- 128 candidates per state;
- 2 frozen predictor families;
- 5 frozen scores;
- 10 predictor/score pairs;
- all Stage A phases unauthorized;
- no models loaded;
- no targets, trajectories, or candidate sets generated;
- no training or fine-tuning;
- all closed resource references verified.

## Do not run

```bash
python experiments/18_planner_score_alignment.py --development-score-audit
```

That command must remain blocked until the validation output and tests are reviewed and a separate one-time authorization is committed.
