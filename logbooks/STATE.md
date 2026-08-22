# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Ranking-aware development no-rerun adjudication  
**Status:** Development completed; raw false integrity traced to heterogeneous-history structural NaNs

## Development outcome

Validation selected ranking weight `1.0` and temperature `0.05` without diagnostic-test input.

| Split | MSE-only retrieval | Ranking-aware retrieval | Gain |
|---|---:|---:|---:|
| Validation | 28.13% | 70.83% | +42.71 points |
| Diagnostic test | 27.08% | 76.04% | +48.96 points |

Diagnostic action-region MSE changed from `0.621324` to `0.623307`. True-versus-intensity improved from 43.23% to 89.06%; true-versus-width improved from 64.58% to 83.85%; position remained 100%.

## Raw integrity issue

The runner's whole-table numeric finiteness check treated method-inapplicable ranking columns on MSE-only history rows as non-finite because Pandas represented the structural blanks as NaN. This is a reporting false positive, not evidence of a non-finite loss.

Every protocol oracle passed, candidates were unique, parameter counts were valid, and tiny-overfit loss decreased. The raw summary is archived unchanged.

## Next action

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/13_ranking_development_adjudication.py
```

Expected: 69 tests. Send `outputs/ranking-aware-latent-development-2026-08-22/development_protocol_adjudication.json`.

Do not rerun development. Formal seeds remain untouched and unauthorized. After adjudication, freeze the selected setting and implement the guarded formal command.
