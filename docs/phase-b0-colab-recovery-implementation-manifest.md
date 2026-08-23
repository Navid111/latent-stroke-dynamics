# Phase B0 Colab recovery runner implementation manifest

## Lifecycle

Status: implemented for validation while recovery remains unauthorized.

This manifest records source implementation only. No renderer transitions, targets, state banks, candidate sets, recovery output, scientific checkpoint, or scientific result was generated.

## Commit sequence

- Recovery protocol freeze: `b14fec2a0702e54d006978fa6e7e794b5566049b`
- Device-aware Phase B training and planning: `a5ba69c41089f162768f0dff7386c734d2a54524`
- Recovery lifecycle guards and initial tests: `22e30554ee46552ceaa192eb1cd8f57934229368`
- Persistent guarded recovery runner: `5f62de00195dfee0358a631937c30487218ea61b`
- Manifest-byte continuity fix and runner tests: `e2561c7d7d658d397d145c81b9c3b0ff605150f7`

## Implemented files

- `configs/phase-b0-colab-recovery-2026-08-24.json`
- `docs/phase-b0-colab-recovery-protocol.md`
- `src/latent_stroke_dynamics/phase_b_recovery.py`
- `src/latent_stroke_dynamics/phase_b_recovery_execution.py`
- `experiments/23_phase_b_colab_recovery.py`
- `tests/test_phase_b_recovery.py`
- `tests/test_phase_b_recovery_runner.py`
- `docs/phase-b0-colab-recovery-command.md`

## Guarded behavior

The recovery command cannot run while the recovery config is frozen and unauthorized. It fails before CUDA environment work, artifact-root creation, renderer-data generation, historical model loading, or training.

Validation-only mode checks the archived zero-completion attempt, passing preflight, fixed environment specification, exact four-manifest continuity hashes, six-resource policy, external persistence policy, stage order, and all later-phase locks. It reports every scientific side effect as false.

## Execution implementation behind the guard

After a later separate authorization only, the runner will:

1. require the exact mounted Google Drive artifact root;
2. require the frozen Tesla T4 software and deterministic float32 environment;
3. verify all six raw resource hashes and loaded model-state hashes;
4. generate the frozen data in a new Drive `.incomplete` directory;
5. verify all four generated JSON manifests byte-for-byte against the interrupted local attempt before the first optimizer step;
6. train the two frozen variants on CUDA;
7. save each completed variant checkpoint and history immediately;
8. run diagnostics and the exact six-method long-horizon comparison;
9. use only the three MSE-only archived predictors, never ranking-aware checkpoints;
10. preserve partial tables and the stage journal on interruption;
11. enforce the original six-hour cap;
12. write the decision and artifact hash manifest;
13. atomically rename the incomplete directory only after completion.

Automatic resume is not authorized.

## Validation handoff

Run locally from the repository root with the virtual environment active:

```bash
git pull --ff-only
python -m pytest
python experiments/23_phase_b_colab_recovery.py --validate-only
```

Expected: `132 passed` and status `phase_b0_colab_recovery_runner_valid_unauthorized`.

A passing local result does not authorize recovery. A new exact bundle and dummy-only Colab validation are still required before a separate one-time recovery authorization.
