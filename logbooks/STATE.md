# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Ranking-aware formal-runner validation  
**Status:** Formal runner implemented; formal data unauthorized

## Frozen development selection

- ranking weight: `1.0`;
- temperature: `0.05`;
- validation retrieval: 70.83%;
- diagnostic retrieval: 76.04%;
- diagnostic gain over MSE-only: 48.96 points;
- development integrity: passed after no-rerun adjudication.

## Formal runner

Implemented:

- frozen prerequisite/hash checks;
- formal/development fingerprint separation;
- matched MSE-only and ranking-aware training across seeds 11/22/33;
- validation-only early stopping;
- one primary test evaluation;
- four secondary stress slices;
- exact conjunctive decision rule;
- method-aware finite history checks;
- oracle, candidate, parameter, overfit, atomic-output, and overwrite guards.

## Next action

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/14_ranking_aware_latent_formal.py --validate-only
```

Expected: 73 tests and status `ranking_latent_formal_runner_valid_unauthorized`. Do not run formal comparison yet.
