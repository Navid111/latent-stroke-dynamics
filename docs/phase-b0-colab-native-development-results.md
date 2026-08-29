# Phase B0 cloud-native development result — 2026-08-24

## Execution completed

The cloud-native Phase B0 experiment completed on a Google Colab Tesla T4. Both models trained on `cuda:0`, all four frozen Linux data-manifest hashes matched, six frozen comparator resources matched, and the outputs finalized atomically in Google Drive. The completed attempt is permanently locked against rerun.

Source commit: `4f0b70dab03f1700a0fbbe5dc9598a1d019b8cc0`

Total wall-clock time: `496.43947252` seconds.

| Variant | Best epoch | Best validation loss | Training time | Diagnostic four-way retrieval |
| --- | ---: | ---: | ---: | ---: |
| Joint prediction only | 40 | 0.0068953314 | 104.49 s | 97.61% |
| Joint prediction + progress | 8 | 0.6048741452 | 83.55 s | 48.16% |

Both representations passed the preregistered non-collapse thresholds. The prediction-only model learned highly discriminative one-stroke transitions. Adding the tested progress and ranking objective substantially weakened four-way retrieval and missed its 50% threshold by 1.84 percentage points.

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
- 128-way regret reduction versus archived MSE-only plus normalized-latent L1;
- mean final-MSE reduction versus that archived baseline;
- no-op parity with joint-prediction-only forced planning;
- maximum final-MSE ratio to exact pixel;
- maximum premature-stop rate.

The progress planner's mean regret was approximately 96.76% worse than the archived latent baseline, and its mean final MSE was approximately 26.52% worse. Its mean final-MSE ratio to exact pixel was 1.935×, above the frozen maximum of 1.5×.

The progress and no-op formulation therefore did not qualify for formal evaluation. The one-time authorization is consumed and the completed run must not be repeated.

## Integrity note — retrospective audit complete

The raw `implementation_integrity` flag remains false because the original strict `transition_splits_disjoint` check counted one shared fingerprint.

A later read-only audit accepted only the canonical cloud-native manifests and found exactly one cross-split fingerprint. That fingerprint matched the analytically derived all-white blank no-op transition. No nonblank, changing, or unknown transition fingerprint crossed a split boundary, and every source manifest remained byte-identical.

This confirms that meaningful changing examples were disjoint across splits while preserving the original strict failure. The audit does not override eligibility, authorize a rerun, or alter the `not_eligible` decision. Multiple independent scientific criteria also failed.

The sanitized audit result is stored in:

- `results/phase-b0-transition-overlap-audit/summary.json`
- `docs/phase-b0-transition-overlap-audit-results.md`

## Interpretation

The experiment answers the model question more precisely:

1. A larger multi-scale action-conditioned joint-embedding model can learn one-stroke consequences extremely well under prediction-only training, reaching 97.61% four-way retrieval.
2. The tested combined progress and ranking supervision did not improve sequential planning; it degraded one-step retrieval and long-horizon decisions.
3. A useful latent-dynamics representation alone is insufficient. Planner-aligned ranking, progress calibration, and stopping remain separate bottlenecks.
4. Formal Phase B0, Phase B1, and Phase B2 remain unauthorized.

## Post-closure qualitative inference

On 2026-08-29, the verified prediction-only checkpoint was used without training or fine-tuning to paint an MNIST-style digit with 100 forced strokes and 128 candidates per step.

The latent planner reached best MSE 0.038312 at step 50 and ended at 0.047404. Exact-pixel greedy reached best MSE 0.022178 at step 99 and ended at 0.022196.

The latent trajectory achieved:

- 14% exact top-1;
- 39% exact top-5;
- mean exact rank 13.6 of 128;
- mean one-step regret 0.000722;
- mean score-to-exact Spearman correlation 0.841.

The predictor therefore learned meaningful broad candidate ordering but lacked the top-of-list precision required for clean repeated control. Four-way one-step retrieval of 97.61% did not guarantee precise selection from 128 candidates.

The trajectory reached its minimum at step 50 and then overpainted. The exact comparator continued improving until step 99, showing that useful candidate strokes remained available. This identifies ranking precision—not candidate absence or output resolution alone—as the primary bottleneck in this qualitative case.

This post-closure inference supports the existing interpretation and does not modify the frozen decision.

The sanitized qualitative result is stored in:

- `results/phase-b0-prediction-only-qualitative-mnist-3/summary.json`
- `docs/phase-b0-prediction-only-qualitative-results.md`

Legacy `recovery_*` field names in the raw handoff come from the reused, previously validated execution engine. The handoff's source commit, experiment ID, execution mode, output root, and lifecycle record identify this as the separate cloud-native experiment—not a recovery of the Mac attempt.
