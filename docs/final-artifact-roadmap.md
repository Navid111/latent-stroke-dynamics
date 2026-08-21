# Final image-to-strokes artifact roadmap

**Start:** 2026-08-21  
**Submission deadline:** 2026-09-24  
**Available time:** approximately 34 days

## Deliverable

A reproducible command-line program that accepts an image and produces a sequential grayscale straight-stroke approximation, including the final canvas, stroke list, progress curve, and animation.

## Phase A — planner foundation (21–25 August)

- Freeze Stage 3 protocol.
- Implement image preprocessing.
- Implement deterministic error-guided candidate generation.
- Implement random and exact-greedy planners.
- Add tests for target conversion, candidate validity, deterministic seeds, and monotonic exact selection.

## Phase B — learned planner (25–29 August)

- Add reproducible pixel-model checkpoint saving.
- Load the checkpoint independently of training.
- Batch candidate predictions on CPU.
- Compare predicted candidate scores with exact-renderer scores.
- Complete a one-target smoke with random, exact, and learned methods.

## Phase C — controlled comparison (29 August–3 September)

- Freeze the six-target command.
- Run six renderer-generated targets with matched budgets.
- Produce MSE/MAE curves, ranking diagnostics, runtimes, final canvases, and animations.
- Record success or failure without retuning against controlled targets.

## Phase D — qualitative image demonstrations (3–7 September)

- Run several user-selected or self-owned images.
- Keep them qualitative because arbitrary photographs are outside the synthetic training distribution.
- Select representative success and failure examples.

## Phase E — thesis assembly (7–16 September)

- Finalize Methods, Results, Discussion, and Limitations.
- Verify literature claims and citations against original PDFs.
- Insert final tables and figures.
- Explain the latent failure, pixel control, and planner result as one coherent decomposition.

## Phase F — buffer and defence (16–23 September)

- Resolve formatting and reproducibility issues only.
- Proofread claims and captions.
- Prepare slides and rehearse the architecture, negative result, pivot, and final artifact.
- Preserve 24 September as the submission deadline rather than an experimentation day.

## Scope protections

- Grayscale only.
- 64×64 only for controlled evaluation.
- Straight-line strokes only.
- One-step greedy planning only.
- No reinforcement learning.
- No multi-step rollout.
- No color, textured brushes, or GAN/diffusion component before the core artifact and thesis are complete.
