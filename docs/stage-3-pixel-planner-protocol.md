# Stage 3 protocol — target-guided pixel-space stroke painter

**Status:** Core protocol frozen before implementation  
**Frozen on:** 2026-08-21  
**Deadline:** 2026-09-24  
**Scope decision:** Explicitly reopened after the paired pixel control succeeded  
**Latent Gate 2 decision:** Remains fail

## 1. Goal

Build a working grayscale demonstration that accepts an image, converts it to a 64×64 target, and constructs an approximation sequentially with straight-line strokes.

At each step, a candidate generator proposes strokes. A planner ranks them, executes one stroke with the exact renderer, observes the true resulting canvas, and repeats.

This stage evaluates whether the successful one-step pixel predictor is useful for target-guided candidate ranking. It does not revive the failed latent predictor.

## 2. Methods

All methods start from the same white canvas, use the same candidate budget, and commit selected actions with the exact renderer.

1. **Random selection:** choose one candidate uniformly from the generated set.
2. **Exact greedy pixel planner:** render every candidate exactly and choose the one with lowest target pixel MSE.
3. **Learned pixel planner:** use the frozen-formulation pixel MLP to predict every candidate’s next canvas, rank by target pixel MSE, then commit the selected stroke with the exact renderer.

The exact greedy method is an upper reference for one-step candidate ranking, not a learned model. Random selection is the minimum planning baseline.

A learned latent planner is excluded because latent Gate 2 failed exact-action retrieval. Exact-renderer latent-objective planning may be added only as an explicitly secondary comparison after the three required methods work.

## 3. Target processing

- Input: any local image supported by Pillow.
- Convert to grayscale.
- Center-crop to a square and resize to 64×64.
- Keep a white starting canvas.
- Quantitative controlled targets use the same renderer and action palette, making them reachable by the primitive.
- Arbitrary uploaded images are qualitative demonstrations, not part of the primary decision.

## 4. Controlled target set

Generate six synthetic target canvases, each containing 20 random strokes, using independent fixed seeds:

```text
20260901, 20260902, 20260903,
20260904, 20260905, 20260906
```

The target generator uses widths `{1, 2, 3, 4}`, values `{0, 32, 64, 96, 128}`, and the existing deterministic renderer.

## 5. Candidate proposal

At every planning step, generate 128 candidates using a deterministic RNG stream fixed by target and step.

- 80% error-guided proposals: sample the stroke midpoint from the current absolute target-error map.
- 20% uniform exploration proposals.
- Sample orientation uniformly.
- Sample normalized length in `[0.10, 0.60]`.
- Sample width from `{1, 2, 3, 4}`.
- Set stroke intensity to the closest allowed palette value to the mean target intensity under the proposed stroke mask.
- Reject actions that leave the current canvas unchanged.

Each method uses the same proposal algorithm and candidate count. Candidate sets may differ after paths diverge because error-guided proposals depend on each method’s current canvas; RNG streams and budgets remain matched.

## 6. Learned planner checkpoint

The paired-control experiment did not save weights. Train one separate, clearly labeled **demonstration checkpoint** using:

- the existing 1,000-example training split (`20260824`);
- the existing 200-example validation split (`20260825`);
- MLP seed `11`, selected because it had the best validation error among the paired-control MLP seeds;
- the frozen `11 -> 64 -> 1` architecture;
- 30 epochs, patience 6, AdamW, learning rate `0.001`, weight decay `0.0001`, and batch size 16.

Save the checkpoint and preprocessing metadata under the ignored `checkpoints/` directory. This is a deployment artifact, not a rerun or replacement of the completed paired evaluation. Do not evaluate or select it on the frozen paired test rows.

## 7. Planning run

For each of the six controlled targets and each required method:

- planning steps: 100 strokes;
- candidate budget: 128 per step;
- commit exactly one stroke per step;
- save the exact canvas after every step;
- save selected stroke parameters and candidate-ranking diagnostics;
- use deterministic target/method seeds.

The learned planner must batch candidate prediction to fit the base M1 MacBook Air. Candidate chunks may be reduced for memory without changing candidate sets or decisions.

## 8. Metrics

### Primary

- final target pixel MSE after 100 strokes;
- final target pixel MAE;
- area under the per-step MSE curve;
- improvement from the white initial canvas;
- learned-versus-random and learned-versus-exact final-error ratios.

### Planning diagnostics

For every learned-planner step, also evaluate the same candidates with the exact renderer and report:

- exact rank of the learned-selected candidate;
- exact top-1 agreement;
- exact top-5 agreement;
- one-step exact regret;
- whether the executed stroke improved the true canvas.

### Practical artifacts

- runtime per method;
- selected stroke sequence as JSON;
- progress CSV;
- final canvas PNG;
- target/final comparison PNG;
- GIF or frame sequence showing line-by-line construction.

## 9. Interpretation rule

The learned pixel planner is considered successful for the controlled target set if:

1. it improves final MSE over the white canvas on all six targets;
2. its mean final MSE is at least 20% lower than random selection;
3. its mean final MSE is no more than 25% worse than exact greedy selection;
4. no implementation or checkpoint-integrity check fails.

Top-1/top-5 agreement and regret are reported diagnostically rather than used to tune the model.

If the learned planner fails, the exact greedy planner still provides the requested image-to-strokes artifact, while the learned-planning failure becomes an additional result. Criteria must not change after controlled outputs are inspected.

## 10. User-facing command

The final interface should support a command of the form:

```bash
python paint.py \
  --target path/to/image.png \
  --method learned \
  --strokes 100 \
  --candidates 128 \
  --output-dir outputs/my-painting
```

The command should save the processed target, final canvas, progress metrics, stroke sequence, and animation frames.

## 11. Execution order

1. Commit this protocol before planner implementation.
2. Add deterministic target preprocessing and candidate generation tests.
3. Implement and smoke-test exact greedy planning first.
4. Add checkpoint training/saving without touching paired test evaluation.
5. Implement learned candidate ranking and exact execution.
6. Run a tiny one-target smoke with all three methods.
7. Freeze the six-target controlled command.
8. Run the controlled comparison once.
9. Run qualitative user-image demonstrations.
10. Write the planning result and finish the thesis.
