# Representation extension — no-rerun adjudication handoff

The single frozen run completed. Do not rerun or retune it.

## Why one summary-only command is needed

The raw runner added a bit-equality condition that was not in the frozen written protocol. Its task-autoencoder oracle retrieval was 100% and candidates were unique, but separately batched candidate-zero encodings differed by `1.7404556274414062e-05`, causing a false global integrity failure. The raw result remains preserved unchanged.

The runner also did not apply the written protocol's at-or-below-35% not-usable precedence to ViT-MAE.

The adjudicator reads only `extension_summary.json`. It does not load a model or dataset, generate data, retrain, evaluate, or recompute a scientific metric.

## Run after pulling

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/11_representation_extension_adjudication.py
```

Expected test total: `54 passed`.

Send back:

```text
outputs/representation-extension-2026-08-22/protocol_adjudication.json
```

Expected written-protocol outcomes:

- global protocol integrity: pass;
- task autoencoder: `average_predictable_but_not_action_usable`;
- ViT-MAE: `not_predictively_usable`;
- no previous decision changes.

Do not execute experiment 10 again.
