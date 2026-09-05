# Current State

**Last updated:** 2026-09-05  
**Branch:** `quadratic-bezier-extension`  
**Current stage:** recovered fixed comparison completed, adjudicated, and archived as text-only evidence  
**Status:** `minor_improvement`; material promotion failed; both execution authorizations consumed

## Read first

- [Completed result and verification scope](../docs/quadratic-bezier-fixed-comparison-results.md)
- [Compact numerical/provenance record](../docs/quadratic-bezier-fixed-comparison-result-record-2026-09-05.json)
- [18-pair public audit copy](../docs/quadratic-bezier-fixed-comparison-paired-metrics-audit-copy-2026-09-05.csv)

## Closed evidence

The formal ranking-aware one-step comparison remains a strong positive result at 74.44% retrieval versus 31.44% for MSE-only.

The controlled long-horizon evidence remains closed. Latent ranking improved over random but missed the exact-pixel ratio criterion. The multi-scale prediction-only model achieved 97.61% four-way retrieval, while qualitative 128-candidate exact top-1 was 14% and the trajectory overpainted after step 50.

The exact-pixel RGB baseline and resolution-by-budget ablation are closed. The quality-priority 128x128/420 setting improved mean common-resolution MSE by 17.97% across the fixed five-target set at a 3.56x compute proxy.

## Completed bounded primitive comparison

The separate comparison used six deterministic procedural targets, 128x128 planning, 512x512 evaluation, 420 accepted strokes, 64 candidates per pool, and seeds 73/137/211. Target hashes, source, tests, environment, and decision rules stayed frozen. No learned model or training was used.

| Quantity | Final record |
| --- | --- |
| Completed runs / pairs | 36 / 18 |
| Original complete units reused / missing units recovered | 17 / 19 |
| Straight mean final 512 RGB MSE | 0.008087940717162071 |
| Curve mean final 512 RGB MSE | 0.008027885089692675 |
| Relative aggregate improvement | 0.7425329829873983% |
| Target means improved | 4 of 6 |
| Dense-scene worsening | 6.518152081149964% |
| Recorded category | `minor_improvement` |
| Material quantitative eligibility | false |
| Required blinded review | false |
| Recorded integrity | passed |

The >=5% overall-improvement and <=5% worst-target-worsening conditions failed. The frozen executable retains a positive, integrity-passing aggregate gain as minor improvement. Both failed performance conditions remain visible; no rule or raw output is rewritten.

Actual candidate renders were 492,608 for straight lines and 492,480 for curves. Saved per-run times sum to 627.0291351449991 and 874.8627017620005 seconds respectively, spanning original and recovery runtimes. The recovery interval alone was 790.493188298 seconds, not the combined attempt duration.

## Historical interruption and recovery

Two identical read-only diagnostics, the second after terminating the original runtime, established the earlier 17-complete/one-partial/18-unstarted state with no final aggregate. No comparative content had been viewed at that point.

The separately validated missing-only recovery preserved the 17 complete units, quarantined `03_organic_silhouette/seed_211/quadratic_bezier`, executed exactly 19 missing units, and finalized one aggregate after 36 runs verified. The original journal's top-level in-progress string is a pre-final snapshot in the pinned code, not a live failure or instruction to resume.

- Frozen scientific runner: `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b`.
- Original authorization: `cc857407ed431c5583fd9e1c02a0ba619a8c187a` — consumed.
- Recovery implementation: `46e0c6396f0425ed84812e8fbeef9ed675ef53e9`.
- Recovery validation evidence: `5cc2e6c98bb58b6ad917b593b97dbd359033fe75` — 217 passed in 59.28s.
- Recovery authorization: `76b6d53bddaaa60880e7c7f1eaffd1392c9ece25` — consumed.
- Execution handoff: `fa965bbf0dba029758180f6b6b3626dfcae241bc` — supplied execution-time record: 217 passed in 57.21s.

Cell 6 reported 453 verified artifact hashes and successful no-required-review numerical handoff. Numerical and A/B mapping exposure has since occurred; descriptive montage observations are not an independent blinded visual validation.

## Verification scope

Read-only aggregation of all 18 transcribed CSV rows reconciled the overall/per-target means, seed coverage, pair ratios, relative changes, candidate totals, and run-time totals. The pasted aggregate sidecar checksum matches the handoff: `23bce3625b9e534d9ef5e2b1affdee105fbbdb59751914ba8f3ac7a8492755f2`.

The 453-artifact byte verification is the notebook's recorded verification. Archival did not independently rehash every original Drive file or uploaded attachment. Public CSV/log files are text audit copies; expected original-artifact digests are not asserted to be their byte hashes. No new experiment or test execution was performed for this archival update.

## Manuscript state

All six ULAB-aligned chapters have complete v0.1 bases in Notion. The 2026-09-03 pre-extension snapshot remains unchanged. The bounded result is integrated into the live methodology, implementation, results/discussion, and conclusions. Exact post-extension word counts, the full parameter/citation audit, final figure layout, and cross-chapter refinement remain. No private manuscript content or private links are included in this public record.

## Next action

Review the archived result through [PR #1](https://github.com/Navid111/latent-stroke-dynamics/pull/1), without treating review readiness as permission to merge. Then continue manuscript, parameter/source, figure, citation, university-format, and defense work.

Never rerun either comparison/recovery Cell 5. Cell 7 is unnecessary for the observed branch. Preserve finalized output `quadratic-bezier-fixed-comparison-v1`, its quarantine, and every earlier closed result. The non-material outcome does not authorize a learned curved-stroke predictor or post-result tuning.
