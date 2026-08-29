# Phase B0 prediction-only qualitative result — 2026-08-29

## Purpose and boundary

This post-closure diagnostic asked whether the completed cloud-native prediction-only checkpoint could guide a multi-step painting of an MNIST-style digit. It was inference only: no training, fine-tuning, model selection, or formal evaluation occurred. The source Phase B0 decision remains `not_eligible`.

The verified checkpoint was `joint_prediction_only_seed73.pt` with artifact SHA-256 `d13124a1becb7a11a84eb973a5d3acf72780f9813de6ed9422432778406155b4` and state SHA-256 `c1402cb94faadb504487d7be5295ae1c95a5f0b41dd8485768c9e20c5ed9e462`.

## Setup

- 64×64 grayscale planning canvas;
- automatic dark-on-white polarity normalization;
- 100 forced strokes;
- 128 candidates per step;
- planner seed `20261001`;
- exact renderer execution after each selected action;
- fixed target encoding and current-canvas re-encoding after each exactly rendered action;
- 512×512 supersampled replay for presentation;
- exact-pixel greedy qualitative comparator under the same proposal settings.

## Results

| Metric | Phase B0 prediction-only latent | Exact pixel greedy |
| --- | ---: | ---: |
| Initial MSE | 0.144356 | 0.144356 |
| Best MSE | 0.038312 | 0.022178 |
| Best step | 50 | 99 |
| Final MSE | 0.047404 | 0.022196 |
| Final MAE | 0.103710 | 0.060518 |
| Improving steps | 63/100 | 88/100 |
| Final MSE improvement from blank | 67.16% | 84.62% |

The latent model's best MSE was 1.727× the exact comparator's best MSE. Its final MSE was 2.136× the exact comparator's final MSE. Its best frame reduced MSE by approximately 73.46% from the blank canvas, but the final frame was about 23.73% worse than its own best frame.

Candidate-ranking diagnostics for the latent trajectory were:

- exact top-1: 14%;
- exact top-5: 39%;
- mean exact rank: 13.6 of 128;
- mean one-step regret: 0.000722;
- mean score-to-exact Spearman correlation: 0.841.

## Interpretation

The strong positive rank correlation shows that the trained latent predictor captured meaningful target-relevant ordering across candidate strokes. The model was not selecting randomly and substantially improved the canvas from blank.

However, long-horizon painting depends on repeatedly choosing near the very top of a 128-way candidate set. Four-way one-step retrieval of 97.61% did not guarantee precise 128-way top selection: exact top-1 was only 14%. Small ordering errors accumulated, the trajectory reached its minimum at step 50, and subsequent strokes caused overpainting.

The exact-pixel comparator produced a more recognizable digit and continued improving until step 99. This confirms that useful candidates remained available and identifies ranking precision—rather than candidate absence or output resolution alone—as the principal bottleneck in this case.

## Thesis conclusion

The defensible conclusion is not that latent prediction failed completely. The prediction-only model learned useful one-stroke dynamics and could guide a recognizable partial trajectory, but it was not sufficiently precise for clean long-horizon target-guided painting.

This strengthens the distinction among one-step prediction, small-set retrieval, large-set candidate ordering, and sequential control.

Machine-readable metrics are stored at `results/phase-b0-prediction-only-qualitative-mnist-3/summary.json`. Generated images and GIFs remain in the preserved Drive output and thesis evidence workspace rather than being treated as formal repository evidence.
