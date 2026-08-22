# Ranking-aware latent follow-up — authorized development handoff

Navid validated `66 passed in 6.74s` and status `ranking_latent_development_runner_valid_unauthorized`. Both frozen hashes, parameter counts, synthetic objective gradient, seed reservations, and output-path guards matched. No follow-up data were generated.

Exactly one development-grid execution is now authorized. Formal data remain unauthorized.

## Run

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/12_ranking_aware_latent_followup.py --validate-only
python experiments/12_ranking_aware_latent_followup.py --run-development-grid
```

After pulling, validation-only status must be:

```text
ranking_latent_development_runner_valid_authorized
```

The development grid trains the matched MSE-only baseline plus six ranking settings, all across seeds 11/22/33. It may take several minutes on the base M1. Keep the machine awake and connected to power.

## Success artifact

Send:

```text
outputs/ranking-aware-latent-development-2026-08-22/development_summary.json
```

## Failure handling

If an exception occurs or the process is interrupted:

1. do not rerun;
2. preserve the complete traceback;
3. preserve `outputs/ranking-aware-latent-development-2026-08-22.incomplete/`;
4. send both before changing anything.

Do not generate formal seeds. Development results may select only lambda and temperature under the frozen validation-only rule.
