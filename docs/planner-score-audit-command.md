# Single authorized planner-score development audit

Validation completed successfully before authorization: 96 tests passed, the guarded status was `planner_score_audit_runner_valid_unauthorized`, every closed resource reference matched, and no models or data were loaded or generated.

Exactly one development score audit is now authorized.

## Run now

```bash
git pull --ff-only
pytest
python experiments/18_planner_score_alignment.py --development-score-audit
```

Expected test count after the authorization commit: **96 passed**.

The authorized command will use only the reserved new development seeds. It will evaluate 72 fixed candidate sets, each containing 128 candidates, across both frozen predictor families and all five frozen scores. It will not train or fine-tune any model.

The run may take a while on CPU. Let it finish without opening a second copy of the command.

## Single-use rule

- Run `--development-score-audit` exactly once.
- Do not rerun it after success.
- If it errors or is interrupted, do not delete or rename the `.incomplete` directory and do not retry. Send the complete terminal output for review.
- Do not run planner development or confirmatory experiments yet.

## Expected artifacts

```text
outputs/planner-score-audit-development-2026-08-23/
├── aggregate_summary.csv
├── candidate_scores.csv
├── per_state_summary.csv
├── run_config.json
├── selection.json
├── state_bank.csv
└── targets/
```

After completion, send the complete terminal output and the contents of `aggregate_summary.csv`, `selection.json`, and `run_config.json` for review.
