# Ranking-aware latent formal comparison — validation handoff

## Frozen formal design

- selected ranking weight: `1.0`;
- selected temperature: `0.05`;
- frozen task-autoencoder checkpoint/statistics;
- three matched MSE-only seeds and three ranking-aware seeds;
- primary 1,000/200/300 train/validation/test transitions;
- four 100-example secondary stress slices;
- formal seeds `20261104`–`20261110`;
- conjunctive 50% retrieval, +10-point matched gain, average-error, crowding, oracle, seed, and integrity criteria.

## Implemented safeguards

The runner verifies the archived development adjudication and selected setting, frozen hashes and sizes, local development fingerprints, formal/development disjointness, validation-only early stopping, test exclusion from selection, method-aware history finiteness, candidate uniqueness, exact-target oracles, finite metrics, parameter counts, tiny-overfit behavior, atomic output handling, and overwrite refusal.

## Validate without generating formal data

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/14_ranking_aware_latent_formal.py --validate-only
```

Expected test total: `73 passed`.

Expected status:

```text
ranking_latent_formal_runner_valid_unauthorized
```

Both formal authorization fields must remain false, both formal output paths must be available, and `formal_data_generated`/`models_trained` must remain false.

Send the complete pytest result and validation JSON. Do not use `--run-formal-comparison` yet. Formal authorization will be committed separately after validation review.
