# Current State

**Last updated:** 2026-08-23  
**Branch:** `main`  
**Current stage:** Controlled latent-planner execution  
**Status:** Exactly one controlled run authorized

## Evidence carried forward

The formal ranking-aware comparison remains closed at 74.44% retrieval versus 31.44% for MSE-only. The one-target planner smoke passed implementation integrity and is permanently closed. It found latent MSE better than latent ranking on that diagnostic target, without changing the frozen controlled protocol.

## Controlled validation

The expanded suite passed all 89 tests. No-data validation returned `latent_planner_controlled_runner_valid_unauthorized`: no models were loaded, no controlled target or planner data was generated, both atomic output paths were free, criteria were frozen, and historical evidence remained unchanged.

## Authorization boundary

Exactly one execution is authorized for the six frozen target/planner seed pairs, five methods, 100 steps, and 128 candidates. No model training, fine-tuning, smoke rerun, protocol change, or second controlled execution is authorized.

## Next action

Pull the authorization commit, rerun the 89-test suite, and execute `python experiments/17_latent_planner_controlled.py --controlled-run` exactly once. Keep the machine awake. Preserve and report the complete terminal output, `aggregate_summary.csv`, `decision.json`, and `run_config.json`.
