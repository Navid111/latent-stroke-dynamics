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

The project is now authorized to begin **Gate 2: deterministic one-step latent prediction**.

## Current scope

- 64×64 grayscale canvases
- One straight-line stroke primitive
- Synthetic, deterministic transitions
- A frozen pretrained encoder
- Global features **and** spatial patch features
- No reinforcement learning
- No multi-step rollout until one-step prediction and ranking work

The initial engineering baseline is `facebook/dinov2-small` because it is easy to run and exposes spatial patch tokens. The CLI accepts a different Hugging Face vision-model name later, so an I-JEPA checkpoint can be tested without rewriting the experiment. Using DINOv2 for this diagnostic does **not** turn the thesis into a DINOv2 thesis.

## Project context for humans and agents

- [`AGENTS.md`](AGENTS.md) — operating rules for coding and research agents
- [`logbooks/STATE.md`](logbooks/STATE.md) — current status, decisions, blockers, and next actions
- [`docs/thesis-plan.md`](docs/thesis-plan.md) — concise research plan, architecture, baselines, timeline, and fallback
- [`docs/gate-1-protocol.md`](docs/gate-1-protocol.md) — frozen Gate 1 design and criteria
- [`docs/gate-1-results.md`](docs/gate-1-results.md) — formal Gate 1 result and interpretation

## Gate 1: embedding sensitivity

The finalized diagnostic:

1. reused the same proposed stroke across nested crowding levels,
2. included tiny-noise, dense pixel-MAE-matched, and sparse support-and-MAE-matched controls,
3. recorded action and changed-pixel metadata,
4. measured all-patch, top-10%, changed-region, and reference-stroke-region distances,
5. quantified spatial localization and lift over random,
6. separated crowding levels in every plot,
7. saved paired win rates in `gate_diagnostics.csv`.

The dense matched control was retained as a stress test. The sparse matched control was the primary comparison because it matched both pixel-change amount and support size while destroying coherent line structure.

The low fixed top-10% win rates under clutter are reported rather than hidden. Sparse random pixels touch about four times as many patch locations as the connected line, so that metric rewards spatial dispersion. The preregistered reference-region and localization criteria passed decisively.

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

To reproduce the frozen formal Gate 1 run:

```bash
python experiments/01_embedding_sensitivity.py \
  --samples 25 \
  --crowding 0 5 15 \
  --batch-size 4 \
  --device cpu \
  --seed 20260819 \
  --output-dir outputs/gate1-formal
```

## Generated Gate 1 outputs

Each run writes:

- `results.csv` — one row per controlled comparison with action metadata
- `aggregate_summary.csv` — mean and standard deviation by condition and crowding
- `gate_diagnostics.csv` — paired control win rates and localization summaries
- `distance_distributions.png` — distance plots faceted by crowding
- `localization_metrics.png` — changed-region and top-k localization diagnostics
- `example_patch_heatmap_crowding_<n>.png` — spatial evidence for each crowding level
- `run_config.json` — reproducibility settings

Generated data, figures, and checkpoints are ignored by Git. Curated snapshots may be copied into `results/` and committed with clear labels.

## Repository layout

```text
.
├── AGENTS.md
├── docs/
│   ├── gate-1-protocol.md
│   ├── gate-1-results.md
│   └── thesis-plan.md
├── logbooks/
│   └── STATE.md
├── experiments/
│   └── 01_embedding_sensitivity.py
├── src/latent_stroke_dynamics/
│   ├── encoder.py
│   ├── gate1.py
│   ├── metrics.py
│   └── renderer.py
├── tests/
│   ├── test_gate1.py
│   └── test_renderer.py
├── results/
├── data/
├── outputs/
├── pyproject.toml
└── README.md
```

## Decision rule

Gate 1 is complete. Gate 2 should train and compare deterministic one-step predictors before any candidate ranking, reinforcement learning, or multi-step planning work begins.
