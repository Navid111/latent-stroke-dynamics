# One-time authorization for interrupted quadratic-Bezier comparison recovery

## Authorization basis

Two identical post-termination diagnostics established a stable preserved state containing 17 valid completed units, one partial unit, and 18 units that never started. The guarded recovery implementation at commit `46e0c6396f0425ed84812e8fbeef9ed675ef53e9` then passed the separate no-output validation: 217 tests passed, all seven frozen runner dependency paths remained unchanged, Google Drive was not mounted, and the unauthorized probe failed before creating output. Validation evidence is archived in commit `5cc2e6c98bb58b6ad917b593b97dbd359033fe75`.

## Authorized action

`configs/quadratic-bezier-recovery-authorization-2026-09-05.json` authorizes exactly one completion of the already-started fixed comparison through `run_quadratic_bezier_recovery.py --execute-recovery` at the validated implementation commit.

The recovery must:

1. confirm the exact stable diagnostic state before mutation;
2. byte-verify and reuse the 17 completed units;
3. hash and quarantine `03_organic_silhouette/seed_211/quadratic_bezier` without deletion or overwrite;
4. execute exactly the 19 missing units in the original frozen schedule;
5. repeatedly verify that all 17 reused units remain byte-identical;
6. aggregate only after all 36 canonical units pass integrity checks;
7. preserve the frozen quantitative rule and blinded-review boundary;
8. produce one completed comparison in total.

## Prohibitions

This authorization does not permit a fresh comparison, reuse of the partial unit as evidence, overwrite of any existing unit, target or setting changes, comparative tuning, early metric or identity reveal, training, a learned model, historical reruns, output publication, or automatic resume after another interruption.

If any precondition differs, the recovery must stop before mutation. If recovery begins and is interrupted, preserve every file and perform a new audit; do not execute the authorization again.
