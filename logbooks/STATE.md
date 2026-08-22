# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Thesis integration  
**Status:** Experimental evidence frozen; full representation extension complete and archived

## Frozen completed evidence

- Gate 1 passed.
- DINOv2 Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval.
- Controlled Stage 3 learned planning succeeded across six synthetic targets.
- MNIST qualitative exact greedy outperformed long-horizon learned pixel planning.
- The single full representation extension completed in 2,353.79 seconds.
- The no-rerun adjudicator passed all 54 tests and written-protocol global integrity passed.

No completed result may be rerun, retuned, or replaced.

## Final representation ladder

| Representation | Four-way retrieval | Interpretation |
|---|---:|---|
| Raw pixels | 100.00% | Full-resolution action identity recoverable |
| Task autoencoder | 37.89% | Average-predictable, not action-usable |
| Frozen DINOv2-small | 27.67% | Historical latent Gate 2 fail |
| Frozen ViT-MAE | 7.11% | Not predictively usable |

Historical anchors use earlier paired seeds and are descriptive.

## Final extension findings

The selected 49,569-parameter task autoencoder improved validation reconstruction by 95.47% over the train-mean-image baseline. Its MLP dynamics improved action-region error by 70.65% versus identity and 68.62% versus mean delta. Retrieval was 37.89%, below the frozen 50% threshold.

ViT-MAE MLP dynamics improved average error by 33.08% and 30.63%, but retrieval was 7.11% and crowding-60 stress performance was 13.69% worse than identity.

The raw summary and protocol adjudication are both archived. The adjudication recomputed no metric and changed no historical decision.

## Active work

1. Update the thesis Results draft with controlled Stage 3, MNIST qualitative limitations, and the final representation ladder.
2. Draft Methods and Discussion around the frozen protocols.
3. Select final tables and figures.
4. Verify literature citations against original PDFs.
5. Prepare a concise defence narrative and reproducibility checklist.

## Boundaries

- Do not run experiment 10 again.
- Preserve all controlled outputs, checkpoint hashes, qualitative failures, raw summaries, and adjudication artifacts.
- Keep the user-facing painter pixel-based.
- New experiments are lower priority than thesis integration and require a separately frozen protocol.
