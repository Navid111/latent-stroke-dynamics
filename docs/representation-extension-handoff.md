# Representation extension — development-smoke handoff

## Completed validation

Navid validated:

- `46 passed` before the training utilities were added;
- frozen foundation status `foundation_valid`;
- deterministic ViT-MAE smoke status `mae_encoder_smoke_passed`;
- feature shape `[2, 196, 768]`;
- exact repeated encoding difference `0.0`;
- positive fixed-stroke signal;
- no extension data generated.

The reported unused decoder keys are expected because the pretrained MAE checkpoint contains a reconstruction decoder while the extension intentionally loads the encoder only.

## Step 1 — update and test the development runner

```bash
git pull --ff-only
source .venv/bin/activate
pytest
```

Expected total: `50 passed`.

## Step 2 — run the matched development smoke once

```bash
python experiments/09_representation_extension_development.py
```

This command is intentionally not configurable beyond output location and CPU thread count. It uses only development seeds `20261020`–`20261022` and trains:

- three autoencoder seeds, selected by validation reconstruction MSE;
- linear and MLP dynamics models with seeds `11`, `22`, and `33` for the selected task latent;
- the same six dynamics models for frozen ViT-MAE features.

It may take several minutes on the base M1 MacBook Air. ViT-MAE weights are already cached. Keep the terminal open and prevent sleep until it completes.

## Actual saved JSON output

The command atomically creates:

```text
outputs/representation-extension-development-smoke/
```

The one file to send back first is:

```text
outputs/representation-extension-development-smoke/smoke_summary.json
```

Unlike the earlier validation commands, this is an actual saved JSON file. The directory also contains reconstruction, dynamics, retrieval, history, config, and split-integrity artifacts.

If the command fails, do not delete or rename anything. Send the traceback and preserve:

```text
outputs/representation-extension-development-smoke.incomplete/
```

Do not run the command a second time and do not generate the frozen primary/stress data yet.
