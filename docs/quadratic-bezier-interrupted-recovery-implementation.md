# Guarded recovery implementation for the interrupted quadratic-Bezier comparison

## Scope

Two identical read-only diagnostics were captured, with the second produced only after the original Colab runtime was disconnected and deleted. The stable state contains 17 valid completed units, one partial unit, and 18 units that never started. No completed output directory, aggregate, or failure record exists. No comparative image or metric was opened.

The partial unit is `03_organic_silhouette/seed_211/quadratic_bezier`. It is not a result and may not be used. Nineteen units therefore require execution: the partial unit from scratch plus the 18 units that never started.

## Recovery design

`src/latent_stroke_dynamics/quadratic_bezier_recovery.py` implements a fail-closed continuation that:

1. validates the target freeze and original one-completed-comparison authorization;
2. verifies that every frozen runner dependency remains byte-equivalent to runner commit `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b`;
3. requires the exact stable file count, byte count, completed-unit set, and partial-unit identity;
4. verifies each completed summary checksum, complete artifact manifest, frozen run configuration, action count, progress monotonicity, target hash, schedule identity, and safety flags;
5. hashes every file in the 17 completed units before mutation and checks those manifests after every recovered unit;
6. hashes and atomically moves the partial unit into an audit quarantine without deleting or overwriting it;
7. executes exactly the 19 missing units in the original target, seed, and condition order using the unchanged frozen per-unit runner;
8. rebuilds the aggregate only after all 36 canonical run directories pass the same checks;
9. preserves the quantitative and identity boundary and emits only a minimal blind-gate handoff;
10. atomically renames the `.incomplete` root to the authorized final output only after the aggregate and its checksum exist.

If recovery is interrupted, automatic resume is forbidden. The current state and all audit files must be preserved for a new inspection.

## Authorization lifecycle

The recovery implementation is deliberately unauthorized. `run_quadratic_bezier_recovery.py --validate-only` does not mount or inspect Drive, create outputs, reveal metrics, open images, train a model, or modify closed evidence.

A separate no-output Colab validation must pass first. Its report and full pytest log must then be archived. Only a later authorization file may permit one recovery of this exact preserved state and implementation commit. That authorization cannot permit a fresh comparison, overwrite a completed unit, overwrite the partial unit, tune a setting, reveal metrics early, or train a model.

## Runtime reporting after recovery

Each of the 36 run summaries retains its exact per-unit runtime. Aggregate condition-runtime totals are therefore reconstructable exactly. The recovery wall-clock interval is recorded separately. A single uninterrupted wall-clock duration cannot be reconstructed after the interruption and must not be invented.
