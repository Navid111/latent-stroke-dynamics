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

## Gate 1: embedding sensitivity

The first experiment compares representations for:

1. identical canvases,
2. tiny pixel noise,
3. adding one stroke,
4. shifting a stroke,
5. changing stroke width,
6. changing stroke intensity,
7. repeating those tests with increasingly crowded canvases.

It records pixel, global-feature, and patch-feature distances and saves a spatial feature-difference heatmap.

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

Run a tiny smoke test first:

```bash
python experiments/01_embedding_sensitivity.py \
  --samples 3 \
  --crowding 0 5 \
  --output-dir outputs/gate1-smoke
```

Then run a more meaningful first experiment:

```bash
python experiments/01_embedding_sensitivity.py \
  --samples 25 \
  --crowding 0 5 15 \
  --output-dir outputs/gate1
```

The first run downloads the pretrained encoder. On a machine with CUDA, the script selects the GPU automatically; otherwise it runs on CPU.

## Generated outputs

Each run writes:

- `results.csv` — one row per controlled comparison
- `aggregate_summary.csv` — mean and standard deviation by condition
- `distance_distributions.png` — global, patch, and pixel distances
- `example_patch_heatmap.png` — where the encoder detected change
- `run_config.json` — reproducibility settings

Generated data, figures, and checkpoints are ignored by Git. Keep final thesis figures by moving selected files into a future tracked `figures/` directory.

## Repository layout

```text
.
├── docs/
│   └── gate-1-protocol.md
├── experiments/
│   └── 01_embedding_sensitivity.py
├── src/latent_stroke_dynamics/
│   ├── encoder.py
│   ├── metrics.py
│   └── renderer.py
├── tests/
│   └── test_renderer.py
├── data/
├── outputs/
├── pyproject.toml
└── README.md
```

## Decision rule

Do not begin the dynamics predictor merely because the calendar says so. Move to Gate 2 only if spatial features respond consistently to controlled stroke changes and remain usable as the canvas becomes crowded. See [`docs/gate-1-protocol.md`](docs/gate-1-protocol.md) for the practical pass/fail checklist.
