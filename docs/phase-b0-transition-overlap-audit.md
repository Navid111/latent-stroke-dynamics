# Phase B0 transition-overlap audit

## Purpose

The completed cloud-native Phase B0 run reported
`transition_splits_disjoint: false`, which caused the strict
`implementation_integrity` criterion to fail. The execution code used set
intersection over complete transition fingerprints, where each fingerprint
contains the current canvas, next canvas, action raster, and no-op flag.

This audit determines whether the overlap is limited to the analytically
identical all-white no-op transition or includes any changing/nonblank
transition. It is retrospective interpretation only. It does not change the
completed result, eligibility decision, or authorization state.

## Safety boundary

The audit:

- reads the three existing transition manifest JSON files;
- requires their exact completed cloud-native SHA-256 values before analysis;
- derives the blank no-op SHA-256 directly from fixed bytes;
- does not regenerate renderer data;
- does not import or load a model;
- does not train or write a checkpoint;
- verifies that all three source manifest hashes are unchanged after reading;
- refuses to write its report inside the source manifest directory;
- does not authorize a rerun, formal Phase B0, Phase B1, or Phase B2.

## Required source files

Use the `data_manifests` directory from the completed cloud-native output. It
must contain:

- `train_transitions.json`
- `validation_transitions.json`
- `diagnostic_test_transitions.json`

The command rejects the historical Mac manifests, compatibility-test files,
and any edited copy because their hashes differ. The planner-supervision
manifest is not part of this audit.

## Local validation

From the repository root with the existing environment active:

```bash
python -m pytest -q
```

## Audit command

Choose a new report location outside the historical output directory:

```bash
python scripts/audit_phase_b0_transition_overlap.py \
  --manifest-dir "/absolute/path/to/data_manifests" \
  --output phase-b0-transition-overlap-audit.json \
  --require-blank-no-op-only
```

The command prints the same JSON written to the report. It exits with status 2
if the observed overlap is not classified as `blank_no_op_only`. It fails before
analysis if the three supplied files are not the exact manifests from the
completed cloud-native execution.

## Interpretation

- `blank_no_op_only`: every cross-split collision equals the independently
  derived fingerprint for an all-white current canvas, identical next canvas,
  zero action raster, and `no_op = true`. Nonblank/changing transitions remain
  disjoint.
- `no_cross_split_overlap`: the supplied manifests are strictly disjoint; check
  that the correct completed cloud-native manifests were supplied.
- `contains_nonblank_or_unknown_overlap`: stop and inspect the listed hashes;
  do not reinterpret the integrity failure as a harmless no-op collision.

Even if the result is `blank_no_op_only`, the original run's strict integrity
criterion remains failed as recorded and the final `not_eligible` decision is
unchanged. The audit only explains the source of the collision for accurate
thesis reporting.
