# Phase B0 one-time Colab recovery execution handoff

## Current status

The persistent execution builder, readiness check, and Colab notebook are implemented while recovery remains unauthorized. The execution builder is expected to fail before packaging until a separate exact authorization record is committed.

## Fixed Drive destination

```txt
/content/drive/MyDrive/latent-stroke-dynamics-phase-b0-recovery
```

The final run directory is:

```txt
phase-b0-joint-embedding-development-2026-08-24-colab-recovery
```

An interrupted run remains in the same root with `.incomplete` appended. Automatic resume is forbidden; preserve it and request an audit.

## Notebook safeguards

- exact authorized bundle status and Git commit verification;
- Tesla T4 requirement;
- 145-test fail-closed suite before Drive mounting;
- exact one-time authorization overlay;
- raw and loaded-state resource checks;
- final/incomplete output absence check;
- explicit `RUN_AUTHORIZED_RECOVERY = False` switch;
- persistent unbuffered Drive console log;
- no second log/run allowed;
- atomic final-output verification;
- downloadable completion handoff containing decision, run config, integrity manifest, and lifecycle journal.

## Pre-authorization local gate

```bash
git pull --ff-only
python -m pytest
python experiments/23_phase_b_colab_recovery.py --validate-only
```

Expected: `145 passed` and the existing unauthorized validation status. The execution-bundle builder remains intentionally unusable until the later authorization commit.

After this gate passes, record the exact handoff commit in the one-time authorization, rerun the same 145 tests, build the authorized execution bundle once, and use only that bundle in the execution notebook.
