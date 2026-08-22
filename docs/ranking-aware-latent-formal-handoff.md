# Ranking-aware latent formal comparison — authorized handoff

## Validation evidence

- `73 passed in 5.94s`;
- status: `ranking_latent_formal_runner_valid_unauthorized`;
- frozen checkpoint/statistics hashes matched;
- selected ranking weight/temperature matched `1.0` / `0.05`;
- formal output and incomplete-output paths were clear;
- no formal data generated and no models trained.

## Authorization

A separate commit now authorizes exactly one formal run. No scientific setting changed.

## Final preflight and run

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/14_ranking_aware_latent_formal.py --validate-only
python experiments/14_ranking_aware_latent_formal.py --run-formal-comparison
```

Expected preflight: `73 passed` and status `ranking_latent_formal_runner_valid_authorized`, with both authorization fields true and both output-path-available fields true.

This is the only authorized formal execution. If the command errors, do not retry or delete the `.incomplete` directory; report the complete traceback. If it completes, do not rerun or retune it. Send `outputs/ranking-aware-latent-formal-2026-08-22/formal_summary.json`.
