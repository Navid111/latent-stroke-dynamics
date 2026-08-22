# Frozen full representation extension — validation handoff

The development smoke is archived and cannot be rerun. Its metrics did not change any scientific setting. The only post-smoke repair was correcting the reported autoencoder parameter count from trainable-after-freeze (`0`) to total (`49,569`).

## Step 1 — pull and test

```bash
git pull --ff-only
source .venv/bin/activate
pytest
```

Expected total: `51 passed`.

## Step 2 — validate only

```bash
python experiments/10_representation_extension_full.py --validate-only
```

Expected status:

```text
full_command_valid
```

The validation must also show:

- autoencoder total parameters `49569`;
- task-autoencoder dynamics: linear `1376`, MLP `19232`;
- ViT-MAE dynamics: linear `598272`, MLP `396800`;
- primary/stress data generated: `false`;
- authorized run started: `false`;
- output directory available: `true`.

Stop after validation and send the pytest result plus the printed JSON. Do not use `--run-frozen-extension` until the validation output is reviewed.

## Later single authorized command

After explicit review, the full command will be:

```bash
python experiments/10_representation_extension_full.py --run-frozen-extension
```

It will generate untouched seeds `20261024`–`20261030`, may take substantially longer than the three-minute development smoke, atomically save `extension_summary.json`, and permanently close the extension. It must never be rerun or retuned.
