# Representation extension — implementation handoff

The protocol and configuration were committed before implementation. No extension split has been generated.

## Step 1 — update and test

```bash
git pull --ff-only
source .venv/bin/activate
pytest
```

Expected total: `46 passed`.

## Step 2 — validate the frozen foundation

```bash
python experiments/08_representation_extension.py --validate-only
```

Expected status: `foundation_valid`. The output must say `extension_data_generated: false`.

## Step 3 — deterministic ViT-MAE smoke

```bash
python experiments/08_representation_extension.py --mae-smoke
```

The first run downloads `facebook/vit-mae-base`; the ordinary unauthenticated Hugging Face warning is harmless. Do not paste credentials into chat. The smoke uses only a blank canvas and one fixed handcrafted stroke, not the frozen extension splits.

Expected conditions:

- status `mae_encoder_smoke_passed`;
- patch grid `[14, 14]`;
- feature shape `[2, 196, 768]`;
- maximum repeat difference at or below `1e-7`;
- finite positive stroke signal;
- model frozen;
- `extension_data_generated: false`.

Do not run a full extension command yet. The autoencoder training and matched development runner will be implemented only after these checks pass.
