# Single authorized planner-score long-horizon development comparison

Validation completed successfully before authorization: 103 tests passed and the runner returned `planner_score_planner_development_runner_valid_unauthorized`. Every frozen resource and the archived MSE-only plus normalized-latent L1 selection were verified, while no model or planner data was loaded or generated.

Exactly one planner-development comparison is now authorized.

## Run now

```bash
git pull --ff-only
pytest
python experiments/19_planner_score_development.py --planner-development
```

Expected test count after the authorization commit: **103 passed**.

The run uses only three reserved new targets and compares:

1. exact pixel;
2. learned pixel;
3. current normalized-latent MSE with a forced 100-step horizon;
4. the frozen normalized-latent L1 winner with a forced 100-step horizon;
5. the same L1 winner with the exact-current-state, zero-margin no-op.

No model training or fine-tuning is authorized. Confirmatory evaluation remains unauthorized.

## Single-use rule

- Run `--planner-development` exactly once.
- Do not open a second copy of the command.
- The deterministic replay checks make this slower than one ordinary five-method run; let it finish.
- Do not rerun it after success.
- If it errors or is interrupted, do not delete or rename the `.incomplete` directory and do not retry. Send the complete terminal output for review.
- Do not run confirmatory evaluation.

## Expected artifacts

```text
outputs/planner-score-audit-planner-development-2026-08-23/
├── aggregate_progress.png
├── aggregate_summary.csv
├── decision.json
├── final_montage.png
├── integrity_by_target.json
├── per_target_summary.csv
├── progress_by_step.csv
├── run_config.json
├── step_diagnostics.csv
└── targets/
```

After completion, send the complete terminal output and upload `aggregate_summary.csv`, `per_target_summary.csv`, `decision.json`, `run_config.json`, and `aggregate_progress.png`.
