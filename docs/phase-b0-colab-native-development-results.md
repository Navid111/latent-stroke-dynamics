# Phase B0 cloud-native development result — 2026-08-24

## Execution completed

The new cloud-native Phase B0 experiment completed on a Google Colab Tesla T4. Both models trained on `cuda:0`, all four frozen Linux data-manifest hashes matched, six frozen comparator resources matched, outputs finalized atomically in Google Drive, and the complete attempt is now permanently locked against rerun.

Source commit: `4f0b70dab03f1700a0fbbe5dc9598a1d019b8cc0`

Total wall-clock time: `496.43947252` seconds.

| Variant | Best epoch | Best validation loss | Training time | Diagnostic four-way retrieval |
| --- | ---: | ---: | ---: | ---: |
| Joint prediction only | 40 | 0.0068953314 | 104.49 s | 97.61% |
| Joint prediction + progress | 8 | 0.6048741452 | 83.55 s | 48.16% |

Both representations passed the preregistered non-collapse thresholds. The prediction-only model learned highly discriminative one-stroke transitions. Adding the progress/ranking objective substantially weakened four-way retrieval and missed its 50% threshold by 1.84 percentage points.

## Long-horizon decision

Raw frozen decision: `not_eligible`.

Passed:

- historical artifacts unchanged;
- representation non-collapse at both scales;
- no-op planner improved every target from blank;
- compute cap.

Failed:

- raw implementation-integrity flag;
- progress-model diagnostic retrieval;
- 128-way regret reduction versus archived MSE-only + normalized-latent L1;
- mean final-MSE reduction versus that archived baseline;
- no-op parity with joint-prediction-only forced planning;
- maximum final-MSE ratio to exact pixel;
- maximum premature-stop rate.

The progress planner's mean regret was about 96.76% worse than the archived latent baseline, its mean final MSE was about 26.52% worse, and its mean final-MSE ratio to exact pixel was 1.935× versus the frozen maximum of 1.5×. The new progress/no-op formulation therefore does not qualify for formal evaluation.

## Integrity note

The raw `implementation_integrity` flag is preserved as false because `transition_splits_disjoint` was false. All four expected cloud manifest hashes matched, all six raw resources matched, ranking-aware models remained unloaded, checkpoint hashes were recorded, and historical results remained unchanged.

The likely source of the overlap is the fingerprint definition for intentionally trivial no-op samples: with crowding zero, current canvas and next canvas are both the same blank image, the action raster is all zero, and the no-op flag is identical. Independently seeded splits can therefore contain an identical blank no-op content fingerprint. This should be confirmed by an offline intersection audit before final thesis wording. It does not authorize changing the original flag, overriding eligibility, or rerunning the experiment. The scientific decision remains `not_eligible` even if this integrity sub-check is later adjudicated as a construction artifact, because multiple independent scientific criteria failed.

## Interpretation

The experiment answers the key model question more precisely:

1. A larger multi-scale action-conditioned joint-embedding model can learn one-stroke consequences extremely well under prediction-only training (97.61% four-way retrieval).
2. The tested combined progress/ranking supervision did not improve sequential planning; it degraded one-step retrieval and long-horizon decisions.
3. A learned latent dynamics representation alone is not enough. Planner-aligned value/progress calibration remains the bottleneck.
4. Formal Phase B0, Phase B1, and Phase B2 remain unauthorized.

Legacy `recovery_*` field names in the raw handoff come from the reused, previously validated execution engine. The handoff's source commit, experiment id, environment execution mode, output root, and lifecycle record identify this as the separate cloud-native experiment—not a recovery of the Mac attempt.
