# Latent Stroke Dynamics

Bachelor's thesis experiments on **action-conditioned latent canvas dynamics for stroke-based rendering**.

The long-term question is whether a planner can choose a stroke by predicting how that stroke will change a canvas in a frozen visual representation space. The project uses explicit gates rather than immediately training a world model.

## Current status

**Gate 1 passed on 2026-08-19.** The frozen DINOv2-small patch representation preserved a controlled stroke in the correct spatial region under blank, moderate, and high synthetic crowding.

**Gate 2 development v2 completed on 2026-08-20.** The selected linear predictor reduced held-out action-region MSE by 57.2% versus identity and 51.4% versus mean delta, with positive improvement at all three crowding levels. Exact four-way counterfactual retrieval was only 22/96 (22.9%), despite verified unique candidates. Gate 2 has therefore **not** passed or failed: the implementation is sound, but the mixed development result needs one no-retraining retrieval decomposition before the formal command is frozen.

See [`docs/gate-2-dev-v2.md`](docs/gate-2-dev-v2.md) for the complete review.

## Current scope

- 64×64 grayscale canvases
- One deterministic straight-line stroke
- Frozen `facebook/dinov2-small` patch tokens
- Action-conditioned one-step residual prediction
- No reinforcement learning
- No target-guided planning or multi-step rollout until Gate 2 passes

## Project context

- [`AGENTS.md`](AGENTS.md) — operating rules
- [`logbooks/STATE.md`](logbooks/STATE.md) — current source of truth
- [`docs/thesis-plan.md`](docs/thesis-plan.md) — research plan and scope
- [`docs/gate-1-protocol.md`](docs/gate-1-protocol.md) — frozen Gate 1 design
- [`docs/gate-1-results.md`](docs/gate-1-results.md) — formal Gate 1 result
- [`docs/gate-2-protocol.md`](docs/gate-2-protocol.md) — Gate 2 protocol and Amendment 1
- [`docs/gate-2-smoke-1.md`](docs/gate-2-smoke-1.md) — engineering smoke review
- [`docs/gate-2-dev-v2.md`](docs/gate-2-dev-v2.md) — larger development review

## Quick start

Python 3.10+ is recommended.

```bash
git clone https://github.com/Navid111/latent-stroke-dynamics.git
cd latent-stroke-dynamics
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell
pip install -e ".[dev]"
pytest
```

## Current next step: retrieval decomposition

Pull and test the lightweight diagnostic:

```bash
git pull
pytest
```

Then analyze the existing development-v2 files:

```bash
python experiments/02b_retrieval_diagnostics.py \
  --input-dir outputs/gate2-dev-v2
```

This command does **not** load DINOv2, re-encode canvases, retrain a model, or use formal data. It writes into:

```text
outputs/gate2-dev-v2/retrieval-diagnostics/
```

The outputs include:

- `retrieval_summary.csv`
- `retrieval_by_crowding.csv`
- `retrieval_by_stroke_width.csv`
- `retrieval_by_stroke_value.csv`
- `retrieval_by_stroke_length.csv`
- `retrieval_diagnostic.json`
- `candidate_selection_distribution.png`
- `pairwise_true_win_rates.png`
- `true_margin_distribution.png`
- `retrieval_by_crowding.png`

Do not add `--formal-run`, and do not generate the amended formal splits yet.

## Gate 2 experiment outputs

The main experiment writes exact configuration, overfit diagnostics, split fingerprints, per-example and aggregate errors, metrics by crowding, retrieval rows, training history, feature caches, and diagnostic plots. Generated outputs and caches are ignored by Git; curated formal artifacts may later be copied into `results/`.

## Decision rule

Only an exact, explicitly marked formal run may receive a Gate 2 decision. It must beat identity and mean delta by the frozen aggregate margin, remain positive at all primary crowding levels, retrieve the true outcome above the frozen threshold, and remain stable across three model seeds. Gate 3 begins only after a recorded Gate 2 pass.
