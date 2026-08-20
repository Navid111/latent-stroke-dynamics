# Latent Stroke Dynamics

Bachelor's thesis experiments on **action-conditioned latent canvas dynamics for stroke-based rendering**.

The long-term question is whether a planner can choose a stroke by predicting how that stroke will change a canvas in a frozen visual representation space. The project begins with the smallest necessary tests rather than immediately training a world model.

## Current status

**Gate 1 passed on 2026-08-19.** A frozen DINOv2-small patch representation reliably preserved the local consequence of one controlled stroke under blank, moderate, and high synthetic crowding.

| Crowding | Reference-region wins vs sparse control | Median localization lift | Median reference enrichment |
|---:|---:|---:|---:|
| 0 | 25/25 (100%) | 12.80× | 2.05× |
| 5 | 24/25 (96%) | 10.24× | 4.95× |
| 15 | 25/25 (100%) | 10.60× | 6.77× |

No-change distances remained below `3.6e-7`. The formal run and full interpretation are archived in [`results/gate1-formal/2026-08-19/`](results/gate1-formal/2026-08-19/) and [`docs/gate-1-results.md`](docs/gate-1-results.md).

**Gate 2 is now implemented and awaiting local validation.** Its protocol was frozen before implementation. The code generates independent deterministic transition splits, caches frozen patch features in float16, runs a tiny overfit diagnostic, trains linear and nonlinear action-conditioned residual predictors, compares them with identity and mean-delta baselines, and evaluates counterfactual outcome retrieval. A smoke-sized run can only report `diagnostic_only`; it cannot declare the gate passed.

## Current scope

- 64×64 grayscale canvases
- One straight-line stroke primitive
- Synthetic, deterministic one-step transitions
- Frozen `facebook/dinov2-small` final-layer patch tokens
- Action-conditioned residual prediction
- No reinforcement learning
- No target-guided planning or multi-step rollout until Gate 2 passes

The CLI accepts a different Hugging Face vision-model name later, so an I-JEPA checkpoint can be tested without rewriting the experiment. Using DINOv2 for this diagnostic does **not** turn the thesis into a DINOv2 thesis.

## Project context for humans and agents

- [`AGENTS.md`](AGENTS.md) — operating rules for coding and research agents
- [`logbooks/STATE.md`](logbooks/STATE.md) — current status, decisions, blockers, and next actions
- [`docs/thesis-plan.md`](docs/thesis-plan.md) — concise research plan, architecture, baselines, timeline, and fallback
- [`docs/gate-1-protocol.md`](docs/gate-1-protocol.md) — frozen Gate 1 design and criteria
- [`docs/gate-1-results.md`](docs/gate-1-results.md) — formal Gate 1 result and interpretation
- [`docs/gate-2-protocol.md`](docs/gate-2-protocol.md) — frozen Gate 2 data, models, metrics, and decision rule

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

## Gate 2 M1 smoke test

After pulling the latest `main`, run the tests first. Then use this deliberately small CPU command on the base-model Apple Silicon MacBook Air:

```bash
python experiments/02_one_step_prediction.py \
  --train-samples 64 \
  --val-samples 16 \
  --test-samples 32 \
  --model-seeds 11 \
  --epochs 8 \
  --patience 3 \
  --encode-batch-size 8 \
  --encode-chunk-size 32 \
  --train-batch-size 8 \
  --encoder-device cpu \
  --train-device cpu \
  --output-dir outputs/gate2-smoke
```

The script encodes in chunks, stores cached features as float16, unloads the frozen encoder, and converts only active training batches to float32. On a retry with identical settings, add `--reuse-cache` to avoid re-encoding.

Do **not** run the formal configuration yet. The formal command will be frozen only after the tests, overfit diagnostic, and smoke artifacts have been inspected.

## Gate 2 outputs

A completed run writes:

- `run_config.json` — exact run settings and formal-eligibility flag
- `overfit_check.json` — tiny-set learning sanity check
- `split_metadata.csv` — sample metadata and fingerprints proving split separation
- `training_history.csv` — per-epoch train and validation losses
- `prediction_metrics.csv` — held-out per-example errors
- `aggregate_metrics.csv` — grouped means and standard deviations
- `counterfactual_retrieval.csv` — true/shifted/width/intensity outcome ranking
- `gate_diagnostics.csv` — baseline improvements, crowding checks, retrieval, and status
- `baseline_improvement.png` — held-out improvement over no-change
- `example_residual_prediction.png` — true residual, predicted residual, error, and action mask
- `cache/*.pt` — ignored float16 feature caches

Generated data, figures, caches, and checkpoints are ignored by Git. Curated formal snapshots may later be copied into `results/` and committed with clear labels.

## Historical Gate 1 reproduction

Gate 1 is frozen and does not need to be rerun during active Gate 2 work.

```bash
python experiments/01_embedding_sensitivity.py \
  --samples 25 \
  --crowding 0 5 15 \
  --batch-size 4 \
  --device cpu \
  --seed 20260819 \
  --output-dir outputs/gate1-formal
```

## Repository layout

```text
.
├── AGENTS.md
├── docs/
│   ├── gate-1-protocol.md
│   ├── gate-1-results.md
│   ├── gate-2-protocol.md
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

Gate 2 must beat both identity and mean-delta baselines by the frozen margin, remain positive at all three crowding levels, retrieve the true counterfactual outcome above the frozen threshold, and remain stable across formal seeds. Only an exact frozen formal run may receive a pass, borderline, or fail decision. Candidate ranking toward a target belongs to Gate 3 and begins only after a recorded Gate 2 pass.
