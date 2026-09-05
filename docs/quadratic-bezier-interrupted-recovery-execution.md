# Authorized interrupted-comparison recovery execution — completed

> **Closed on 2026-09-05.** The single authorized recovery completed 36 runs / 18 pairs, reusing 17 completed units and executing 19 missing units. The result is `minor_improvement`, not material improvement. Both execution authorizations are consumed. Never rerun either execution Cell 5. Cell 7 is unnecessary for this observed no-required-review branch. See [the completed result](quadratic-bezier-fixed-comparison-results.md). The procedure below is retained only as historical provenance.

## Exact pins

- frozen scientific runner: `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b`;
- validated recovery implementation: `46e0c6396f0425ed84812e8fbeef9ed675ef53e9`;
- recovery validation evidence: `5cc2e6c98bb58b6ad917b593b97dbd359033fe75`;
- one-time recovery authorization: `76b6d53bddaaa60880e7c7f1eaffd1392c9ece25`.

## Historical procedure — do not execute again

1. Disconnect and delete the no-output validation runtime after its three files are safely downloaded.
2. Open `notebooks/quadratic_bezier_interrupted_recovery_execution.ipynb` in a fresh CPU runtime.
3. Run code cells 1–4 in order. Cell 4 mounts Drive and re-verifies the exact stable state without mutation.
4. Run Cell 5 once. Do not interrupt it. If the browser disconnects or shows reconnecting, do not press Stop; reconnect later and allow the existing runtime to continue.
5. Run Cell 6 only after Cell 5 reports successful finalization.
6. If Cell 6 says blinded review is required, return only:
   - `quadratic_bezier_recovery_blind_handoff.json`;
   - `blinded_review_montage.png`;
   - `blinded_review_sheet.csv`.
   Do not run Cell 7 or open the mapping, aggregate, metrics, plots, or logs.
7. If Cell 6 says no blinded review is required, return the downloaded numerical and audit handoff files.

## Historical failure boundary

If any cell fails, preserve every Drive file and report the exact error. Do not delete, rename manually, repair, restart, resume, or rerun the recovery. Another interruption requires a new read-only audit and authorization decision.

## Scientific boundary and completed outcome

The recovery remained the single fixed comparison. It reused the 17 verified completed units, quarantined the partial unit byte-for-byte, and executed only 19 missing units. It did not change the frozen runner, targets, schedule, seeds, conditions, budgets, thresholds, or blind gate.

The supplied execution-time suite recorded 217 passed in 57.21s. Cell 6 reported 453 verified artifact hashes, integrity passed, and no required blinded review. Mean final 512 MSE improved by 0.7425%, but the dense scene worsened by 6.5182%; material promotion failed. Preserve the completed directory and quarantine. The raw journal's in-progress label is an unmodified pre-final snapshot, not a reason to resume. No tests or painting runs were re-executed for the text-only archival update.
