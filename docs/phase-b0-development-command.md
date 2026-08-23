# Phase B0 one-time development execution

## Current lifecycle

The complete runner passed 116 tests and validation with no side effects. A separate authorization record now permits exactly one Phase B0 development execution. Formal B0, Phase B1, and Phase B2 remain unauthorized.

Authorization file:

```text
configs/phase-b0-development-authorization-2026-08-23.json
```

## Sync and authorization-phase tests

```bash
git pull --ff-only
pytest
```

Expected: `116 passed`.

## Single authorized command

Keep the Mac connected to power. From the repository root, run:

```bash
caffeinate -dimsu python experiments/21_phase_b_development.py --development
```

The runner will generate the frozen new data, train the two preregistered variants, evaluate diagnostics, and run the six-method three-target long-horizon comparison. The total frozen compute cap is six hours.

## Critical one-run rules

- Run the development command once only.
- Do not change any seed, model, objective, threshold, or budget.
- Do not launch a second terminal execution.
- Do not delete or rename either the final or `.incomplete` output directory.
- If the command fails, stop and report the complete traceback. Preserve the `.incomplete` directory for review.
- If it succeeds, preserve all artifacts and send the complete final decision JSON and terminal summary.
- Do not tune against the result.
- Do not run any formal, B1, or B2 experiment.

Expected output root:

```text
outputs/phase-b0-joint-embedding-development-2026-08-23
```
