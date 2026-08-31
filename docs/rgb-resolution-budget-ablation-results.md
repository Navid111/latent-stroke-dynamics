# RGB resolution x stroke-budget ablation result

## Status

The one preregistered run completed successfully on 2026-08-31. The archived
96x96/210-stroke baseline was verified and reused; only conditions B, C, and D
were executed. The run did not train or load a learned model and did not alter
the frozen Phase B0 decision.

The machine-generated aggregate intentionally left `final_decision` null until
visual review. All five fixed montage rows have now been inspected. Condition D
passes that qualitative check, so the external final adjudication is:

**Final decision: `meaningful_improvement`; selected condition: D (128x128
planning, 420 strokes).**

The private raw output and machine-generated aggregate remain immutable. This
document records the required post-run visual adjudication without rewriting
them.

## Integrity

- Runner status: `rgb_resolution_budget_ablation_complete`
- Protocol: `rgb_resolution_budget_ablation_v1`
- Validated implementation: `da1f10ceb5b0501479f0037da40b00ccb8ad122b`
- Target-set SHA-256: `31e1fcc2bf344f8b72d3f04dfbc9109c61c39fc8cbc10668c1b78d575a673b42`
- Archived baseline summary SHA-256: `7f2c32cef077bef1737a3f00ee584cf1075b1feb5711995ab87dd9812f233c05`
- Archived baseline artifacts verified: 94
- New conditions completed: 3 of 3
- New artifact hashes verified: 287
- Total runtime: 476.842613864 seconds on a standard Google Colab CPU runtime
- Training performed: no
- Learned model used: no
- Phase B0 changed: no
- Source images and generated binary outputs committed: no

Every new condition completed all five targets, preserved all frozen decisions,
and accepted only strokes that improved its exact planning objective. Best
matched final for every target.

## Primary quantitative result

The primary cross-resolution metric was mean RGB MSE against independently
resized common 512x512 targets. Lower is better. The preregistered eligibility
threshold was a mean of at most `0.011087133841861078` (10 percent below the
archived baseline) with no target more than 5 percent worse.

| Condition | Planning | Strokes | Mean 512 MSE | Ratio to A | Mean reduction | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 96x96 | 210 | 0.012319037602 | 1.000000 | baseline | no (reference) |
| B | 96x96 | 420 | 0.011248509852 | 0.913100 | 8.69% | no |
| C | 128x128 | 210 | 0.011662944922 | 0.946742 | 5.33% | no |
| D | 128x128 | 420 | 0.010105467295 | 0.820313 | 17.97% | **yes** |

Only D crossed the fixed mean threshold. It also improved every target relative
to A:

| Target category | A MSE | D MSE | D reduction |
| --- | ---: | ---: | ---: |
| Simple symbol | 0.010000524173 | 0.008309552854 | 16.91% |
| Geometric composition | 0.028554800606 | 0.024022469741 | 15.87% |
| Isolated object | 0.003459385812 | 0.002835628280 | 18.03% |
| Landscape | 0.008596450471 | 0.005949092026 | 30.80% |
| Dense scene | 0.010984026948 | 0.009410593575 | 14.32% |

D's worst target ratio to A was `0.8567525935091679`, so there was no hidden
per-target regression.

## Factor reading

- Doubling only the stroke budget (B versus A) reduced mean MSE by
  `0.001070527750` but narrowly missed the 10 percent decision threshold.
- Raising only planning resolution (C versus A) reduced mean MSE by
  `0.000656092680`.
- Combining both (D versus A) reduced mean MSE by `0.002213570307`.
- The interaction term was `-0.000486949877`, indicating that the combined
  reduction was larger than the sum of the two isolated reductions.

Within these fixed settings, stroke budget had the larger isolated effect, but
resolution and budget worked best together. D used a 3.56x compute proxy versus
A, so this is a measured quality-cost trade-off rather than a free improvement.

## Qualitative review

The fixed montage was inspected row by row before final adjudication.

1. **Simple symbol:** D retains a coherent circle and three-spoke structure and
   reduces the fragmented appearance of A. Curves remain jagged because the
   renderer still uses straight opaque lines.
2. **Geometric composition:** D preserves the large colored regions, black
   circles, and grid arrangement at least as well as A while giving denser
   boundary coverage. Exact rectangular and circular edges remain faceted.
3. **Isolated object:** D preserves the vase silhouette, narrow neck, pale upper
   body, and blue lower region clearly; no visual regression is apparent.
4. **Landscape:** D gives the clearest qualitative gain, with stronger layered
   hills, horizon placement, and sky structure.
5. **Dense scene:** D modestly improves tonal and vertical structure, but the
   architecture remains highly abstract. This confirms the straight-line
   primitive bottleneck rather than invalidating the configuration gain.

No row showed an obvious qualitative regression that would override the
quantitative decision. D therefore passes the required visual review.

## Interpretation and scope

The defensible conclusion is narrow: for this fixed exact-pixel painter and
five preregistered RGB targets, jointly increasing planning resolution from 96
to 128 and the accepted-stroke budget from 210 to 420 produced a meaningful and
consistent improvement under the frozen rule. This is an engineering result for
the renderer/planner configuration, not evidence that a learned latent model
improved.

D is the quality-priority configuration for demonstrations. A remains the
immutable lower-cost baseline. The result does not solve smooth curves, crisp
geometry, fine architecture, semantic planning, or learned long-horizon action
selection; those belong to the post-defense extension roadmap.

No post-result tuning, target replacement, or rerun is authorized. Experimental
development is frozen after this result so thesis synthesis can resume.

## Publication boundary

The five source images and all generated binary outputs remain private and
uncommitted. The public repository records only method, metrics, hashes, and the
qualitative conclusion. Reproducing the private montage or target images in a
submitted thesis requires a verified licence, institutional rights basis, or
replacement with original/reusable material.
