# Phase B0 cloud recovery — currently unauthorized

## Current lifecycle

The initial local development attempt was interrupted with Ctrl+C during the first variant because of a thermal concern. The audit found only four deterministic data-manifest JSON files, no checkpoint, no training history, no long-horizon target, no final output, and zero completed executions.

The original authorization is consumed and recovery is locked. Formal B0, Phase B1, and Phase B2 remain unauthorized.

## Preservation rule

Do not delete, rename, edit, or use:

```text
outputs/phase-b0-joint-embedding-development-2026-08-23.incomplete
```

Its hashes are archived in:

```text
configs/phase-b0-aborted-local-attempt-2026-08-23.json
```

## Allowed next work

Only a random-dummy-tensor cloud preflight may be prepared and run. It may verify:

- CUDA availability and actual device use;
- package and hardware information;
- frozen resource file and model-state hashes;
- CPU/GPU numerical tolerance on deterministic dummy tensors;
- dummy-only throughput.

It may not generate renderer transitions, targets, state banks, candidate sets, checkpoints, or scientific outputs.

## Prohibited

Do not run `experiments/21_phase_b_development.py --development` locally or in the cloud. A separately validated and committed recovery protocol is required before any new development execution.
