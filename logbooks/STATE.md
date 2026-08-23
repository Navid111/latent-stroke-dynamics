# Current State

**Last updated:** 2026-08-23  
**Branch:** `main`  
**Current stage:** Controlled latent-planner runner validation  
**Status:** Runner implemented; controlled comparison unauthorized

## Evidence carried forward

The formal ranking-aware comparison remains closed at 74.44% retrieval versus 31.44% for MSE-only. The one-target planner smoke passed implementation integrity and is permanently closed. It found latent MSE better than latent ranking on that diagnostic target, without changing the frozen controlled protocol.

## Guarded controlled runner

The runner covers six reserved target/planner seed pairs, five methods, 100 steps, and 128 candidates. It loads frozen models only after authorization, generates targets only inside an atomic `.incomplete` output, preserves best/final states and step diagnostics, repeats learned trajectories for deterministic replay, validates method-aware metrics, aggregates without model selection, and applies only the criteria frozen before data.

## Authorization boundary

Controlled authorization remains false. Validation-only must load no model, generate no target, and create no output. The completed smoke remains unauthorized and cannot be rerun. No training or fine-tuning is authorized.

## Next action

Pull the runner commit, run the expanded test suite, then execute `python experiments/17_latent_planner_controlled.py --validate-only`. Preserve the complete output. Do not run `--controlled-run`.
