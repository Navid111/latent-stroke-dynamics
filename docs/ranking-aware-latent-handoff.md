# Ranking-aware latent follow-up — validation handoff

The protocol and config were committed before implementation and before any follow-up data generation. The foundation contains only strict configuration/hash validation and synthetic ranking-loss tests.

## Run now

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/12_ranking_aware_latent_followup.py --validate-only
```

Expected test total: `62 passed` if every new test is collected. If pytest reports a different passing total, send the complete summary rather than assuming failure.

Validation-only mode may load the already-completed frozen checkpoint and saved latent-statistics JSON. It does not generate a transition, encode follow-up data, train a model, or create an output directory.

## Send back

Send the complete pytest result and printed validation JSON. The important new field is:

```text
latent_statistics_file_sha256
```

Expected validation status before that hash is committed:

```text
ranking_latent_foundation_valid_hash_freeze_required
```

Development must remain unauthorized and both output directories must remain available.

## Stop boundary

Do not generate development data yet. After reviewing the validation output, the exact saved statistics hash will be frozen in a separate commit. Only then can the development runner be implemented and authorized.
