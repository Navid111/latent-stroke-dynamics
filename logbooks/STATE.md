# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Full representation extension complete  
**Status:** Raw result archived; pure written-protocol adjudication pending local validation

## Frozen completed evidence

- Gate 1 passed.
- DINOv2 Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval.
- Controlled Stage 3 learned planning succeeded across six synthetic targets.
- MNIST qualitative exact greedy outperformed long-horizon learned pixel planning.
- The single full representation extension completed in 2,353.79 seconds.

No completed result may be rerun, retuned, or relabeled without preserving the raw record.

## Full-extension result

The task autoencoder passed reconstruction eligibility with 95.47% validation improvement over the train-mean image baseline. Its selected MLP improved action-region error by 70.65% versus identity and 68.62% versus mean delta, but retrieval was 37.89%.

ViT-MAE's selected MLP improved average action-region error by 33.08% and 30.63%, but retrieval was only 7.11% and crowding-60 stress performance was 13.69% worse than identity.

## Raw-run integrity status

The raw runner marked global integrity false because the task exact-target oracle required both 100% top-1 and exact zero difference between separately batched candidate-zero encodings. Top-1 was 100%, all candidates were unique, and the maximum difference was `1.7404556274414062e-05`. Bit equality was not a frozen written-protocol requirement.

The raw runner also failed to prioritize the written protocol's at-or-below-35% not-usable rule for ViT-MAE. Both raw labels remain archived unchanged.

## Next action

```bash
git pull --ff-only
source .venv/bin/activate
pytest
python experiments/11_representation_extension_adjudication.py
```

Expected: 54 tests. Send `outputs/representation-extension-2026-08-22/protocol_adjudication.json`.

This command reads only the completed summary. It performs no data generation, model load, training, evaluation, or metric recomputation.

## Expected written-protocol result pending validation

- global integrity: pass;
- task autoencoder: average-predictable but not action-usable;
- ViT-MAE: not predictively usable;
- raw pixels remain the strongest action representation;
- all historical decisions remain unchanged.
