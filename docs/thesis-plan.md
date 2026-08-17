# Thesis plan — action-conditioned latent canvas dynamics for stroke-based rendering

## Working title

**Painting with Strokes, Predicting in Latent Space: A Feasibility Study of Action-Conditioned Canvas Dynamics**

## One-sentence aim

Test whether a frozen visual representation can support prediction of one-stroke canvas transitions and whether those predictions can help rank candidate strokes toward a target image.

## Motivation

Stroke-based rendering constructs an image as a sequence of parameterized marks. Existing systems demonstrate many ways to generate or optimize strokes, including reinforcement learning, differentiable rendering, direct prediction, and transformer-based methods. Separately, latent world models predict how states change under actions, while joint-embedding methods learn or use representations without reconstructing every pixel.

This project studies a narrow intersection: treat the canvas as the state, a stroke as the action, and the rendered next canvas as the transition. Instead of requiring a learned model to decode a full next image, predict the next canvas in a frozen visual feature space and test whether that prediction is useful for selecting strokes.

The intended contribution is a careful **feasibility study and empirical decomposition**, not a claim to have invented stroke-based rendering, world models, or JEPA. The components have prior art; the thesis asks whether their combination works under a small, controlled digital-painting setup.

## Research questions

### RQ1 — Representation sensitivity

Do frozen global and spatial visual features reliably distinguish canvases that differ by one controlled stroke? Do they preserve where the change occurred, and does the signal survive as the canvas becomes crowded?

### RQ2 — One-step predictability

Given the current canvas representation and a parameterized stroke, can a small deterministic model predict the representation of the true rendered next canvas better than trivial and linear baselines?

### RQ3 — Planning utility

Can predicted next representations rank sampled candidate strokes in a way that improves progress toward a target image compared with random selection and relevant exact-renderer baselines?

### RQ4 — Optional short-horizon value

If one-step selection works, does depth-2 or depth-3 receding-horizon planning improve results enough to justify its computation and accumulated prediction error?

RQ4 is a stretch question, not a requirement for a successful bachelor's thesis.

## Formal setup

Let:

- `C_t` be the current canvas,
- `a_t` be a parameterized stroke,
- `R` be the deterministic renderer,
- `E` be a frozen visual encoder,
- `F_θ` be the learned action-conditioned dynamics predictor,
- `T` be a target image.

The true transition is:

```text
C_(t+1) = R(C_t, a_t)
```

The encoder produces current and next representations:

```text
z_t     = E(C_t)
z_(t+1) = E(C_(t+1))
```

The learned model predicts the consequence of the proposed stroke:

```text
z_hat_(t+1) = F_θ(z_t, a_t)
```

A basic dynamics objective is:

```text
L_dyn = distance(z_hat_(t+1), stop_gradient(z_(t+1)))
```

For target-guided one-step candidate selection:

```text
a* = argmin_a distance(F_θ(E(C_t), a), E(T))
```

The planner proposes candidate strokes. The predictor estimates their consequences. The predictor does not independently generate “the next stroke.” After selecting an action, the system should render it with `R`, encode the real resulting canvas, and plan again.

## Initial scope

To finish within approximately one month:

- Canvas: 64×64 initially; 128×128 only if inexpensive.
- Colour: grayscale first.
- Primitive: one straight-line stroke with endpoints, width, and intensity.
- Data: synthetically generated one-step transitions.
- Renderer: deterministic and treated as ground truth.
- Encoder: one frozen pretrained model at a time.
- Features: spatial patch or intermediate features, with a global feature as comparison.
- Predictor: small deterministic one-step residual model.
- Planner: sampled candidate ranking.
- Rollout: depth 1 first; depth 2–3 optional.
- No reinforcement learning unless all core experiments are complete and there is a specific justified need.

## Stage 1 — Gate 1: representation sensitivity

Create before/after pairs that differ by one controlled factor:

- no change,
- tiny pixel noise,
- adding a stroke,
- shifting its position,
- changing its width,
- changing its intensity,
- increasing the number of prior strokes.

Compare global and patch-feature distances. Inspect distributions across many pairs and spatial heatmaps, not isolated examples.

### Gate 1 pass condition

Proceed when meaningful stroke changes are consistently separated from controls, patch changes are spatially meaningful, and sensitivity remains usable on moderately crowded canvases. The practical initial criterion is documented in `docs/gate-1-protocol.md`.

### Gate 1 fallback

Try a justified intermediate layer, alternate frozen encoder, thicker primitive, or modest resolution change. If multiple sensible configurations still discard one-stroke information, pivot to a representation-suitability thesis rather than forcing a dynamics model.

## Stage 2 — Gate 2: deterministic one-step prediction

Generate tuples:

```text
(current canvas, stroke action, next canvas)
```

Encode the current and true next canvases with the selected frozen feature extractor. Train a small model conditioned on the current representation and action. Begin with residual prediction where appropriate:

```text
z_hat_(t+1) = z_t + delta_θ(z_t, a_t)
```

Possible action encodings should be compared only as needed:

1. a vector of normalized stroke parameters,
2. a spatial action mask rendered separately,
3. a combination of both.

### Required prediction baselines

- No-change predictor: `z_hat_(t+1) = z_t`.
- Mean-delta predictor.
- Linear action-conditioned model.
- Small nonlinear deterministic predictor.

### Gate 2 measurements

- Held-out next-representation error.
- Improvement over the no-change and linear baselines.
- Error by stroke width, intensity, position, and canvas crowding.
- Spatial error maps.
- Candidate-ranking agreement with exact encoded outcomes.

### Gate 2 pass condition

The learned predictor must beat trivial baselines on held-out transitions and preserve enough relative accuracy to rank candidate actions. Low average feature error alone is insufficient if action rankings are wrong.

## Stage 3 — Gate 3: target-guided stroke selection

At each step:

1. Encode the current canvas and target.
2. Sample a fixed candidate set of strokes.
3. Predict each candidate's next representation.
4. Rank candidates by target distance.
5. Commit the best stroke with the exact renderer.
6. Re-encode the true canvas and repeat.

Begin with simple synthetic targets or grayscale images where results are interpretable. Keep candidate budgets equal across methods.

### Core comparison matrix

| Method | Transition evaluation | Selection objective |
|---|---|---|
| A | Exact renderer | Pixel distance |
| B | Exact renderer | Frozen latent distance |
| C | Learned latent predictor | Frozen latent distance |

Also include random candidate selection. This matrix distinguishes:

- whether the latent objective itself is useful (`A` versus `B`),
- whether the learned predictor approximates exact latent ranking (`B` versus `C`),
- and whether any method beats chance.

### Evaluation

Do not use the same frozen encoder as both the planning objective and the sole judge of final quality. Report a combination of:

- pixel MAE or MSE,
- SSIM if appropriate,
- target-distance curves over stroke count,
- candidate-ranking correlation or top-k agreement,
- predictor error by rollout depth,
- runtime or candidates evaluated per second,
- qualitative sequences and failure cases.

A perceptual metric may be added later, but it should not replace image-space evaluation for these simple controlled canvases.

## Optional Stage 4 — short-horizon planning

Only after one-step ranking works, compare depth 1 with depth 2 and possibly depth 3. Use receding-horizon control: plan several hypothetical strokes, execute only the first, observe the real rendered result, and replan.

Measure whether additional depth improves final image quality enough to offset extra computation and model error. A negative result is acceptable and informative.

## Four-week execution plan

### 18–23 August — representation and data pipeline

- Set up the environment and verify tests.
- Run the Gate 1 smoke test.
- Run the controlled sensitivity experiment.
- Compare global and spatial features.
- Select one encoder feature configuration or document the pivot.
- Begin writing the method and Gate 1 result while details are fresh.

### 24–30 August — one-step predictor

- Generate train, validation, and test transitions with fixed seeds.
- Implement trivial, linear, and small nonlinear predictors.
- Evaluate held-out prediction and ranking accuracy.
- Decide whether Gate 2 passes.

### 31 August–6 September — candidate selection

- Implement exact-renderer pixel and latent baselines.
- Implement predicted-latent candidate ranking.
- Produce controlled sequential paintings.
- Attempt depth-2 planning only if one-step selection is stable.

### 7–14 September — final experiments and writing

- Freeze configurations.
- Run final comparisons and ablations.
- Produce tables, figures, and failure analyses.
- Complete method, experiments, results, and limitations chapters.

### 15–23 September — buffer and defence preparation

- Repair only experiment-critical problems.
- Finish introduction and conclusion.
- Check citations and claims.
- Prepare slides and rehearse the explanation of the architecture, novelty, and limitations.

## Fallback ladder

The project has useful stopping points:

1. **Representation result:** characterize whether frozen features preserve incremental stroke changes.
2. **Prediction result:** show whether next-canvas features can be predicted from the current features and action.
3. **Planning result:** show whether predictions improve candidate-stroke selection.
4. **Short-horizon result:** evaluate whether deeper latent rollout helps.

Reaching level 2 with sound baselines and analysis is a legitimate bachelor's thesis. Level 3 is the desired result. Level 4 is optional.

## Related-work map

Use the full literature notes when writing formal citations. The immediate conceptual map includes:

- Stroke generation and planning: *Paint Transformer*, *Learning to Paint*, *SPIRAL*, *Neural Painters*, differentiable stroke-planning work, and *IMPASTO*.
- Latent dynamics and planning: *World Models* and *PlaNet*.
- Joint-embedding visual representations: *I-JEPA* and *V-JEPA*.
- Structured or action-conditioned drawing representations: *CoSE* and adjacent sketch-generation work.

Verify author names, venues, years, and exact claims against the original PDFs before including them in the thesis.

## Claims to avoid

- Do not claim a new JEPA architecture if using a frozen encoder with an action-conditioned predictor.
- Do not claim the model predicts the next stroke when it predicts the next canvas representation.
- Do not claim “first ever” based only on the local paper collection.
- Do not claim stochastic dynamics are necessary without an ablation.
- Do not claim successful planning based only on decreasing distance in the same feature space used for selection.
- Do not hide negative results or scope changes.

## Definition of a successful submission

A successful thesis should contain:

- a reproducible renderer and transition generator,
- an explicit frozen-feature diagnostic,
- appropriate baselines for every attempted stage,
- at least one clearly answered research question,
- quantitative results and interpretable figures,
- documented negative results and limitations,
- a restrained novelty statement,
- and enough implementation detail for another student to reproduce the experiment.
