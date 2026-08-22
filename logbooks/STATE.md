# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Ranking-aware formal execution  
**Status:** One formal run authorized after 73-test validation

## Authorization evidence

- pytest: `73 passed in 5.94s`;
- validation status: `ranking_latent_formal_runner_valid_unauthorized`;
- formal data/models before authorization: none;
- output and incomplete-output paths: clear;
- checkpoint, statistics, development adjudication, and development metadata hashes recorded;
- selected setting: lambda `1.0`, temperature `0.05`.

## Next action

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/14_ranking_aware_latent_formal.py --validate-only
python experiments/14_ranking_aware_latent_formal.py --run-formal-comparison
```

Expected preflight: 73 tests and status `ranking_latent_formal_runner_valid_authorized`.

This run is one-shot. On error, preserve `.incomplete` and send the complete traceback. On success, send `outputs/ranking-aware-latent-formal-2026-08-22/formal_summary.json` and do not rerun.
