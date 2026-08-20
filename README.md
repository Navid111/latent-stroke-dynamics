# Latent Stroke Dynamics

Bachelor's thesis experiments on **action-conditioned latent canvas dynamics for stroke-based rendering**.

The long-term question is whether a planner can choose a stroke by predicting how that stroke will change a canvas in a frozen visual representation space. The project begins with the smallest necessary tests rather than immediately training a world model.

## Current status

**Gate 1 passed on 2026-08-19.** A frozen DINOv2-small patch representation reliably preserved the local consequence of one controlled stroke under blank, moderate, and high synthetic crowding. The formal run and interpretation are archived in [`results/gate1-formal/2026-08-19/`](results/gate1-formal/2026-08-19/) and [`docs/gate-1-results.md`](docs/gate-1-results.md).

**Gate 2 engineering smoke 1 completed on 2026-08-20.** The end-to-end M1 path worked, the tiny-overfit check reduced loss by 98.14%, and the selected linear model showed promising aggregate action-region error. The run also exposed duplicate counterfactual outcomes and accidental use of formal-seed prefixes. Both integrity issues were repaired before any formal run; the original smoke remains recorded in [`docs/gate-2-smoke-1.md`](docs/gate-2-smoke-1.md).

Gate 2 has **not** passed or failed. Development v2 is next, and the amended formal data remain untouched.

## Current scope

- 64×64 grayscale canvases
- One straight-line stroke primitive
- Synthetic, deterministic one-step transitions
- Frozen `facebook/dinov2-small` final-layer patch tokens
- Action-conditioned residual prediction
- No reinforcement learning
- No target-guided planning or multi-step rollout until Gate 2 passes

## Project context for humans and agents

- [`AGENTS.md`](AGENTS.md) — operating rules for coding and research agents
- [`logbooks/STATE.md`](logbooks/STATE.md) — current status, decisions, blockers, and next actions
- [`docs/thesis-plan.md`](docs/thesis-plan.md) — concise research plan and scope
- [`docs/gate-1-protocol.md`](docs/gate-1-protocol.md) — frozen Gate 1 design
- [`docs/gate-1-results.md`](docs/gate-1-results.md) — formal Gate 1 interpretation
- [`docs/gate-2-protocol.md`](docs/gate-2-protocol.md) — Gate 2 protocol and transparent Amendment 1
- [`docs/gate-2-smoke-1.md`](docs/gate-2-smoke-1.md) — first engineering-smoke review

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

## Gate 2 development v2

Pull the latest code and run the tests before starting:

```bash
git pull
pytest
```

Then run the larger but still M1-safe development check:

```bash
python experiments/02_one_step_prediction.py \
  --train-samples 256 \
  --val-samples 64 \
  --test-samples 96 \
  --model-seeds 11 \
  --epochs 30 \
  --patience 6 \
  --encode-batch-size 8 \
  --encode-chunk-size 32 \
  --train-batch-size 16 \
  --encoder-device cpu \
  --train-device cpu \
  --output-dir outputs/gate2-dev-v2
```

This uses permanently development-only seeds. It validates the corrected counterfactual set, provides less noisy crowding and retrieval estimates, and reveals validation convergence. It must report `diagnostic_only`.

On an identical retry, add `--reuse-cache` to avoid re-encoding. Do **not** add `--formal-run`, and do not run the formal sizes or amended formal seeds yet.

## Gate 2 outputs

A completed run writes:

- `run_config.json` — exact settings, candidate validity, and formal eligibility
- `overfit_check.json` — tiny-set learning sanity check
- `split_metadata.csv` — sample metadata and split fingerprints
- `training_history.csv` and `training_curves.png`
- `prediction_metrics.csv`
- `aggregate_metrics.csv`
- `aggregate_metrics_by_crowding.csv`
- `counterfactual_retrieval.csv` and `counterfactual_retrieval.png`
- `gate_diagnostics.csv`
- `baseline_improvement.png`
- `crowding_improvement.png`
- `example_residual_prediction.png`
- `cache/*.pt` — ignored float16 feature caches

Generated data, figures, caches, and checkpoints are ignored by Git. Curated formal snapshots may later be copied into `results/` and committed with clear labels.

## Repository layout

```text
.
├── AGENTS.md
├── docs/
│   ├── gate-1-protocol.md
│   ├── gate-1-results.md
│   ├── gate-2-protocol.md
│   ├── gate-2-smoke-1.md
│   └── thesis-plan.md
├── logbooks/
│   ├── STATE.md
│   └── 2026-08-20.md
├── experiments/
│   ├── 01_embedding_sensitivity.py
│   └── 02_one_step_prediction.py
├── src/latent_stroke_dynamics/
│   ├── encoder.py
│   ├── gate1.py
│   ├── gate2.py
│   ├── metrics.py
│   └── renderer.py
├── tests/
│   ├── test_gate1.py
│   ├── test_gate2.py
│   └── test_renderer.py
├── results/
├── data/
├── outputs/
├── pyproject.toml
└── README.md
```

## Decision rule

Only an exact, explicitly marked formal run may receive a Gate 2 decision. It must beat identity and mean delta by the frozen margin, remain positive at all primary crowding levels, retrieve the true outcome above the frozen threshold, and remain stable across three model seeds. Candidate ranking toward a target belongs to Gate 3 and begins only after a recorded Gate 2 pass.
