# Straight lines versus quadratic Bezier curves — fixed comparison results

**Adjudicated:** 2026-09-05  
**Protocol:** `quadratic_bezier_extension_v1`  
**Completion:** `verified_interrupted_attempt_recovery`  
**Final decision:** `minor_improvement` — not material improvement

## Scope and public evidence

This is a bounded exact-pixel RGB renderer-and-proposal comparison, not a learned painter, a JEPA experiment, or a test of either primitive's theoretical maximum approximation capacity. The repository owner approved this text-only publication. No source images, generated binaries, private manuscript text, or private attachment URLs are included.

Companion records:

- [Compact numerical, hash, and source record](quadratic-bezier-fixed-comparison-result-record-2026-09-05.json)
- [All 18 paired metrics — public text audit copy](quadratic-bezier-fixed-comparison-paired-metrics-audit-copy-2026-09-05.csv)
- [Execution-time pytest log — text copy](quadratic-bezier-recovery-execution-pytest-2026-09-05.txt)
- [Execution completion log — text copy](quadratic-bezier-recovery-execution-log-2026-09-05.txt)
- [Frozen protocol](quadratic-bezier-extension-protocol.md)
- [Historical recovery execution handoff, now closed](quadratic-bezier-interrupted-recovery-execution.md)

The original complete output, aggregate, sidecar, per-run manifests, images, mapping, journal, and quarantined partial unit remain preserved outside the repository. Public CSV/log copies preserve the supplied text values; their line endings or serialization need not be byte-identical to the originals. Expected original-artifact hashes below and in the JSON record apply to the original output, not these public text copies.

## Frozen design

- Six original deterministic procedural targets: ring symbol, curved glyph, organic silhouette, mixed geometry, layered landscape, and dense city scene.
- Seeds 73, 137, and 211; two conditions: `straight` and `quadratic_bezier`.
- 18 target-seed pairs / 36 condition runs.
- 128x128 planning and 512x512 replay/evaluation.
- 64 candidates per pool; 80% error-guided / 20% uniform proposals.
- Global/structure/detail budgets: 80/140/200 accepted strokes, maximum 420.
- Minimum planning improvement `1e-9`; stage patience 12 non-improving pools.
- Opaque, constant-width strokes with target-fitted RGB color; curves add one control point.
- No learned model, training, variable width, transparency, texture, erasing, cubic curves, or mixed primitives.

All supplied run summaries report 420 accepted strokes and best step 420. The six targets are different from the earlier five-target RGB set; the two studies' absolute mean errors are not a same-benchmark comparison.

## Primary result and frozen decision

The primary statistic is the ratio of the two overall mean final 512 RGB MSE values across all 18 pairs, not the mean of percentage changes.

| Quantity | Value |
| --- | --- |
| Straight mean final 512 RGB MSE | 0.008087940717162071 |
| Quadratic Bezier mean final 512 RGB MSE | 0.008027885089692675 |
| Curve / straight ratio of means | 0.992574670170126 |
| Relative mean MSE reduction | 0.7425329829873983% |
| Target means improved | 4 of 6 |
| Maximum target-mean worsening ratio | 1.0651815208114996 |
| Dense-scene mean worsening | 6.518152081149964% |

| Material criterion | Outcome |
| --- | --- |
| At least 5% overall mean improvement | Failed |
| At least four of six target means improve | Passed |
| No target mean more than 5% worse | Failed: dense scene |
| Integrity | Passed in the recorded Colab verification |
| Required blinded review | Not triggered: quantitatively ineligible |

The frozen `evaluate_quantitative_decision` implementation classifies a positive aggregate gain with integrity passed but incomplete material eligibility as `minor_improvement`. A failed performance guard is not silently relabelled as an integrity failure. Preserve both failed performance conditions and the original machine category; no threshold or raw output was edited retrospectively.

## All-target means

Each target mean covers seeds 73, 137, and 211. Negative relative change is better. The 5% worsening guard is applied to each three-seed target mean, not the worst individual seed.

| Target | Straight mean 512 MSE | Curve mean 512 MSE | Curve / straight | Change |
| --- | --- | --- | --- | --- |
| 01_ring_symbol | 0.013228925409756601 | 0.01285966803341196 | 0.9720871223544502 | -2.7913% |
| 02_curved_glyph | 0.004990925202024828 | 0.0045249296125859885 | 0.9066314219155631 | -9.3369% |
| 03_organic_silhouette | 0.0026317239979961465 | 0.0026615854788276717 | 1.01134673729246 | +1.1347% |
| 04_mixed_geometry | 0.007308243373421247 | 0.006787996449776676 | 0.9288136838003214 | -7.1186% |
| 05_layered_landscape | 0.0014793143671037318 | 0.0012134370759426796 | 0.8202699189073661 | -17.9730% |
| 06_dense_scene | 0.018888511952669867 | 0.02011969388761108 | 1.0651815208114996 | +6.5182% |

The landscape's relatively large percentage gain starts from a small baseline error; the dense scene contributes a larger absolute regression. Changing to a mean of target percentages would change the scientific question and was not done.

## Compute and runtime accounting

| Quantity | Straight | Quadratic Bezier |
| --- | --- | --- |
| Condition runs | 18 | 18 |
| Accepted strokes per run | 420 | 420 |
| Actual candidate renders | 492608 | 492480 |
| Sum of saved run runtimes, seconds | 627.0291351449991 | 874.8627017620005 |

Saved run times span the original and recovery runtimes and include planning/replay/per-run artifact work. They are descriptive measurements, not a hardware-normalized benchmark. Matching settings did not require equal rasterization cost or exactly equal realized proposal pools.

The recovery interval alone was `790.493188298` seconds. The wall clock of the interrupted original attempt is not reconstructed by adding or substituting these values.

## Interruption and preservation-safe completion

The stable starting state contained 215 files / 6,958,862 bytes: 17 valid completed runs, one partial run, and 18 unstarted runs. Two read-only diagnostics agreed, including the diagnostic after the original runtime was terminated. At that stage no completed aggregate or comparative exposure existed.

Recovery reused the 17 completed runs without overwrite, quarantined the partial `03_organic_silhouette/seed_211/quadratic_bezier` directory, and executed exactly 19 missing units in frozen order. After all 36 verified runs existed, it constructed the 18 pairs and one final aggregate and atomically finalized output `quadratic-bezier-fixed-comparison-v1`.

The journal retains the original top-level `quadratic_bezier_recovery_in_progress` string. Inspection of the pinned source shows that this field is initialized once and not updated by the event appender. Its final event is `aggregate_ready_for_atomic_finalize`; the aggregate and separate completion handoff establish final completion. Do not repair the raw journal or treat it as permission to resume.

| Source role | Commit |
| --- | --- |
| Validated renderer/replay | `7bdcd2e847ca7c5a1faf8a086b26441d8de1a4e1` |
| Target freeze | `bcbc3221612a891ffaf3dbeb7743d7f365597ce5` |
| Frozen scientific runner | `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b` |
| Original execution authorization | `cc857407ed431c5583fd9e1c02a0ba619a8c187a` |
| Recovery implementation | `46e0c6396f0425ed84812e8fbeef9ed675ef53e9` |
| Recovery validation evidence | `5cc2e6c98bb58b6ad917b593b97dbd359033fe75` |
| Recovery authorization | `76b6d53bddaaa60880e7c7f1eaffd1392c9ece25` |
| Recovery execution notebook/handoff | `fa965bbf0dba029758180f6b6b3626dfcae241bc` |

The earlier recovery validation recorded 217 passed in 59.28s. The supplied execution-time suite recorded **217 passed in 57.21s**. These are different records. No tests or experiment were re-executed for this documentation-only archival commit.

## Verification and visual-evidence limits

Cell 6 reported 453 verified artifact hashes, integrity passed, and a numerical handoff with no required blinded review. The supplied execution log records no training, no learned model, and no changes to closed experiments.

The pasted aggregate sidecar checksum agrees with the handoff:

```text
23bce3625b9e534d9ef5e2b1affdee105fbbdb59751914ba8f3ac7a8492755f2
```

Read-only SQL aggregation of an 18-row transcription of the supplied CSV confirmed six targets, three distinct seeds per target (73/137/211), all reported means, candidate totals, and saved run-time totals. Maximum absolute discrepancies in the supplied pair ratios and relative changes were zero. This is an independent arithmetic consistency check, not independent rehashing of the complete Drive output or all uploaded bytes. The compact JSON distinguishes those scopes.

The original three plots and 18-pair montage remain in the private evidence archive. The first two plots concern final 512 RGB MSE. The progress plot concerns 128x128 planning MSE versus accepted stroke, not a 512-resolution trajectory. Strict acceptance guarantees improvement at planning resolution; it does not prove monotonic improvement of every hypothetical intermediate high-resolution replay.

Numerical results were exposed before montage inspection. Descriptive observations preceded consultation of the A/B mapping, but no independent blinded preference test, visual promotion pass, or perceptual significance result is claimed. No binary figure is published here.

## Conclusion and next work

Close the single comparison as `minor_improvement`; retain the straight-line 128x128/420 quality-priority configuration. The result does not authorize a learned curved-stroke predictor. It does not establish statistical equivalence or show that Bezier primitives are generally ineffective: the outcome is bounded to this proposer, finite budget, procedural set, and seed schedule.

Both authorizations are consumed. Never rerun either execution Cell 5, recreate the incomplete output, replace targets, add seeds, tune parameters, or alter the original aggregate/quarantine. The earlier learned-model and RGB results remain unchanged. Continue manuscript/source/citation/figure refinement and defense preparation; review of this archive does not authorize a PR merge.
