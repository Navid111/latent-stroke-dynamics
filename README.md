# Latent Stroke Dynamics

Bachelor's thesis experiments on **action-conditioned latent canvas dynamics for stroke-based rendering**.

The long-term question is whether a planner can choose a stroke by predicting how that stroke will change a canvas in a frozen visual representation space. The project starts with the smallest necessary test instead of immediately training a world model:

> Does a frozen visual encoder reliably notice one controlled brushstroke, including where and how it changed the canvas?

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
- [`docs/gate-1-protocol.md`](docs/gate-1-protocol.md) — final controlled design and frozen gate criteria

## Gate 1: embedding sensitivity

The finalized diagnostic:

1. reuses the same proposed stroke across nested crowding levels,
2. includes tiny-noise, dense pixel-MAE-matched, and sparse support-and-MAE-matched controls,
3. records action and changed-pixel metadata,
4. measures all-patch, top-10%, changed-region, and reference-stroke-region distances,
5. quantifies spatial localization and lift over random,
6. separates crowding levels in every plot,
7. saves a compact `gate_diagnostics.csv` with paired win rates.

The dense matched control remains a stress test. The sparse matched control is the primary comparison because it matches both pixel-change amount and support size while destroying coherent line structure.

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

Final-control M1 CPU smoke test:

```bash
python experiments/01_embedding_sensitivity.py \
  --samples 3 \
  --crowding 0 5 \
  --batch-size 4 \
  --device cpu \
  --output-dir outputs/gate1-v3-smoke
```

After that smoke test verifies matching and output structure, the formal Gate 1 run is:

```bash
python experiments/01_embedding_sensitivity.py \
  --samples 25 \
  --crowding 0 5 15 \
  --batch-size 4 \
  --device cpu \
  --output-dir outputs/gate1-formal
```

The experimental design and criteria must remain frozen during the formal run.

## Generated outputs

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

Do not begin the dynamics predictor merely because the calendar says so. Move to Gate 2 only if the paired separation and localization criteria in [`docs/gate-1-protocol.md`](docs/gate-1-protocol.md) are met or a deviation is explicitly justified and documented.
