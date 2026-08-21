# Paired pixel-control visual review — 2026-08-21

This review uses the final PNG artifacts together with the saved CSV metrics. Numerical tables remain authoritative.

## Baseline improvement

The mean-delta baseline improves by only about 4.7%. The linear predictor improves by about 92%, while the validation-selected MLP and exact oracle both visually reach 100%. The MLP is actually at 99.950%, but that small difference from the oracle is not visible at this scale. The 30% reference is correctly labeled as a reference rather than a per-slice requirement.

For a thesis figure, accompany this plot with the numerical table because the near-ceiling MLP and oracle bars look identical.

## Improvement by crowding

Mean delta remains near 4–5% at every crowding level. Linear performance decreases from roughly 96% on blank canvases to 88% at crowding 15. MLP remains visually at 100% throughout, consistent with the exact values of 99.973%, 99.955%, and 99.907%.

The legend partially covers the upper-left plotting region near the first group. This is cosmetic. If the figure is regenerated for the thesis, move the legend outside the axes.

## Counterfactual retrieval

Identity and mean delta are near zero, linear reaches 96.1%, and both MLP and exact oracle reach 100%. The 25% random-choice reference and 50% threshold are visible. No bars are ambiguous. This plot is the clearest direct visual contrast with latent Gate 2, where learned-model retrieval remained near chance.

## Validation curves

All six training curves decrease. Linear seeds begin at substantially different losses and converge gradually to roughly 0.03–0.04. MLP seeds decrease by several orders of magnitude and approach zero by the final epochs. Seed 22 remains somewhat above seeds 11 and 33, matching its higher final pixel MSE, but it still achieves perfect retrieval.

There is no visual evidence of divergence, late validation deterioration, or seed collapse. A logarithmic y-axis would make late MLP differences easier to see, but the current plot is sufficient for stability checking.

## Example pixel prediction

The current canvas is crowded. The true next canvas adds a shallow descending stroke across the upper portion. The predicted next canvas is visually indistinguishable from the rendered target at ordinary scale and preserves the stroke’s location, thickness, and intensity.

The absolute-error panel is automatically color-normalized and has no colorbar. It therefore visually amplifies very small residuals and must not be interpreted as showing large canvas-wide error. The saved test MSE is the quantitative authority. For the thesis, either:

1. show only current, true next, predicted next, and action mask; or
2. retain the error panel with an explicit caption stating that it is auto-scaled for visibility.

Do not claim pixel-perfect identity from the image; the prediction has tiny but nonzero error.

## Conclusion

The visual artifacts agree with the tabular paired-control result. The learned MLP is stable, nearly exact in pixel error, perfect in four-way action retrieval, and robust to crowding. There is no visual sign of an implementation contradiction. The experimental core is complete.
