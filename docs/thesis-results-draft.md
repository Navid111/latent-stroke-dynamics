# Thesis results chapter draft

> Draft status: technically grounded from frozen experiment records. Add final chapter numbering, figure numbers, and literature citations during thesis assembly.

## 1. Overview

The experiments evaluated three successive questions. First, whether frozen visual features preserve the effect of a single rendered stroke. Second, whether a small action-conditioned model can predict the next frozen representation. Third, after latent action ranking failed, whether the same deterministic transition is recoverable in a full-resolution pixel formulation.

The results form a mixed but coherent feasibility finding. Frozen DINOv2 spatial features responded to local stroke changes, and a learned latent predictor substantially reduced held-out average error. However, the latent prediction was not precise enough to identify the exact action among closely related counterfactuals. A paired pixel-space control recovered exact actions perfectly, showing that the underlying deterministic transition and action information were learnable.

## 2. Representation sensitivity

Gate 1 evaluated frozen global and spatial DINOv2 features on controlled before-and-after canvases. The spatial patch features preserved meaningful one-stroke changes in the action-aligned region, including under moderate canvas crowding. This supported using the final-layer patch-token grid as the state target for one-step dynamics. Global pooling was not selected because it diluted localized changes.

Gate 1 therefore passed its frozen criterion. This result established representation sensitivity, but it did not establish that the full next representation could be predicted accurately enough for action selection.

## 3. Latent one-step prediction

### 3.1 Model selection and integrity

The latent experiment used independently generated train, validation, test, and stress splits. DINOv2-small remained frozen. Identity, mean-delta, shared linear, and shared MLP residual predictors were compared. The model family was selected using validation action-region MSE, and the MLP was selected before test interpretation.

All formal integrity checks passed. Counterfactual outcomes were unique in pixel and encoded space, all metrics were finite, the implementation overfit check reduced loss by 97.7%, and every learned-model seed beat identity. The formal run was eligible under the frozen configuration.

### 3.2 Average prediction error

On the held-out test split, identity obtained action-region MSE of 0.002250 and mean delta obtained 0.002003. The selected MLP’s three-seed mean was 0.000860, corresponding to improvements of 61.8% over identity and 57.1% over mean delta.

The MLP result was stable across initialization seeds: 0.000861, 0.000841, and 0.000877. Improvement over identity remained positive at all primary crowding levels, although it decreased from 79.0% at crowding 0 to 43.3% at crowding 5 and 25.0% at crowding 15. Secondary unseen-width, unseen-intensity, and unseen-crowding slices also remained better than identity.

These results show that broad one-step latent consequences were learnable and generalized beyond one initialization or one in-distribution condition.

### 3.3 Exact-action retrieval

Average error was not the only criterion. For each test transition, the predicted next representation was compared with four exact rendered outcomes: the true action, a shifted action, a width-changed action, and an intensity-changed action. Retrieval counted as correct only when the true result was closest to the prediction. Four-way random choice gives a 25% reference.

The selected latent MLP achieved only 27.7% retrieval, below the frozen 50% requirement and within the protocol’s fail region. Seed results ranged from 26.3% to 29.3%, so the failure was stable rather than caused by one initialization.

Pairwise decomposition localized the problem. The true result beat shifted-position outcomes 77.9% of the time and changed-intensity outcomes 75.2% of the time, but beat changed-width outcomes only 40.7% of the time. Width-changed candidates were selected 48.2% of the time. Increasing the development data to the full formal set improved average prediction and other action dimensions but left width discrimination essentially unchanged.

Gate 2 therefore formally failed under its conjunctive rule. The failure was scientific rather than an execution or implementation failure: low latent MSE did not guarantee exact action identity.

## 4. Paired pixel-space control

### 4.1 Purpose and formulation

The explanatory control tested whether the deterministic transition could be learned when predicting normalized full-resolution pixels. It reused the paired train, validation, and test split seeds. Per-pixel inputs contained the current pixel value, the same seven-value stroke vector, an exact binary proposed-stroke mask, and normalized coordinates. The selected MLP contained 833 parameters and predicted a residual pixel value.

The control also included a renderer-equivalent exact compositing oracle. The oracle achieved effectively zero error and 100% retrieval, validating candidate order, mask construction, reconstruction, and retrieval scoring.

### 4.2 Pixel prediction result

Validation selected the MLP. Test action-region MSE was 0.000249, compared with 0.499396 for identity and 0.475834 for mean delta. This corresponds to improvements of 99.950% and 99.948%, respectively.

All three MLP seeds remained near exact. Their test action-region MSE values were 0.000178, 0.000376, and 0.000193. Improvement over identity was 99.973%, 99.955%, and 99.907% at crowding 0, 5, and 15. Errors also remained very small for unseen width, intensity, and crowding slices.

### 4.3 Pixel retrieval result

Every MLP seed retrieved the true outcome for all 300 paired test transitions. Mean four-way retrieval was therefore 100%, with zero seed-level spread. Pairwise true-result win rates for position, width, and intensity were all 100%.

The linear pixel model also achieved 96.1% retrieval despite substantially larger pixel reconstruction error. This observation is informative: candidate ranking can remain correct despite imperfect average reconstruction. In the latent experiment, the opposite occurred—average error was strong while exact ranking remained poor.

The pixel control succeeded under its frozen interpretation rule. Exact action information was recoverable in the tested full-resolution pixel formulation.

## 5. Latent-versus-pixel interpretation

The paired comparison rules out several broad explanations for latent failure. The deterministic renderer was functioning correctly, counterfactual candidates were distinct, the action contained sufficient position, width, and intensity information, the dataset size was sufficient for the basic transition, and a tiny deterministic model could learn the mapping consistently.

The remaining bottleneck lies in the overall tested latent patch formulation. Possible mechanisms include frozen-feature invariance, coarse 16×16 tokenization, fractional action-mask downsampling, normalized latent MSE, independent patch-wise prediction, or regression smoothing. The experiments do not isolate one mechanism because the pixel control changes target space, spatial resolution, and mask resolution together.

Accordingly, the result does not support the broad claim that DINOv2 or latent prediction is inherently unsuitable. Gate 1 showed that DINOv2 patch features notice local strokes, and latent average prediction generalized meaningfully. The narrower supported claim is that the tested frozen DINOv2 patch-token predictor lacked the exact action precision available in the tested full-resolution pixel formulation.

## 6. Answer to the research question

The action-conditioned world-model framing is feasible at a controlled one-step level: a compact model can learn deterministic stroke consequences from current state and action. However, the specific frozen latent formulation evaluated here is not sufficiently precise for exact candidate-stroke ranking, despite low held-out latent error. Full-resolution pixel prediction provides a strong positive control and exposes the gap between average state prediction and planning-relevant action discrimination.

Target-guided latent planning was therefore not attempted. Under the frozen project logic, proceeding to planning with a predictor that failed exact-action retrieval would risk attributing planner failure to the wrong component.

## 7. Limitations and future work

The study used synthetic 64×64 grayscale canvases and straight-line strokes. The pixel model received an exact full-resolution action mask and therefore learned a relatively simple compositing operation. Its 100% retrieval result must not be generalized to curved brushes, textured media, color images, larger canvases, real photographs, or target-guided painting.

Future work could separately test higher-resolution latent features, intermediate encoder layers, spatially interacting predictors, contrastive or ranking-aware objectives, and width-sensitive losses. Such work should use new data splits and be labeled as follow-up rather than replacing the frozen negative result. Multi-step planning should be attempted only after a latent or alternative predictive representation demonstrates stable one-step action ranking.
