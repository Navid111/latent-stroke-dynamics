# Gate 2 formal visual review — 2026-08-21

This review uses the formal PNG artifacts together with the exact CSV metrics. Plots support interpretation; CSV values remain authoritative.

## Baseline improvement

Both trainable families clearly exceed the aggregate 30% margin. Linear improves by roughly 60% and the validation-selected MLP by 61.8%. Mean delta improves only about 11%. The bars are separate and unobstructed.

## Improvement by crowding

The learned models are strongest on blank canvases and weaken monotonically as prior-stroke crowding rises. MLP is slightly stronger than linear at crowding 0, while linear is slightly stronger at 5 and 15. Mean delta becomes worse than identity at crowding 5 and 15.

The learned models remain positive at crowding 15, satisfying the frozen slice rule. The horizontal 30% line can be misleading on this particular plot: 30% was the aggregate margin, while the frozen per-crowding requirement was positive improvement. A thesis-facing regenerated plot should remove that line or label it as aggregate reference only; this is a presentation correction, not a scientific change.

## Counterfactual retrieval

Identity and mean delta remain far below chance. Linear is slightly below the 25% random reference. MLP is only slightly above it at 27.7%, with a large visual gap to the 50% threshold. No bars overlap. This plot makes the sole gate failure unambiguous.

## Validation curves

All three linear seeds fall rapidly and then plateau near `0.00069`. The three MLP seeds begin worse but continue improving, cross below the linear family late in training, and finish near `0.00063`–`0.00065`. Curves overlap closely within families, with no divergence or seed collapse. Validation-based selection of MLP is visually consistent with the saved metrics.

## Residual example

The proposed stroke is a shallow diagonal segment across the upper portion of the patch grid. The true residual has a bright, spatially extended response along that segment. The predicted residual places its strongest response in the same broad area but is dimmer, narrower, and less spatially complete on the shared scale. The error map remains brightest along much of the proposed stroke.

This directly supports the mixed result: the model localizes the broad consequence but underestimates action-specific magnitude and extent. A single example cannot prove that width is the dominant confusion, so that claim must come from the full retrieval decomposition rather than the heatmap alone.

## Conclusion

The visual artifacts agree with the tabular formal result. There is no plotting evidence of instability, seed collapse, candidate aliasing, or a hidden metric contradiction. Strong average-error prediction and weak exact retrieval coexist coherently.
