# Phase B0 Colab recovery — one-time authorization

## Authorization decision

Exactly one cloud recovery execution is authorized for the previously interrupted zero-completion Phase B0 development attempt.

This authorization is based on:

- validated scientific runner source `2c38ffeee6a1182153cfed65fbcd1ece9f357781`;
- validated persistent execution handoff `3191e3e6b382bea96bf48569f3ac5af3eec61b24`;
- 145 passing local tests on the complete pre-authorization handoff;
- passing Tesla T4 dummy-only implementation status `phase_b0_colab_recovery_implementation_valid_unauthorized`;
- validation bundle SHA-256 `2d5d8ab7c15d33f72d4d4db7b69e7b96a903fa33881f712a1cf6433969bd7138`;
- exact six-resource and loaded-state integrity checks;
- zero completed variants and zero scientific decisions from the interrupted local attempt.

## Authorized scope

- one completed Phase B0 recovery execution;
- exact frozen architecture, losses, seeds, budgets, methods, thresholds, and six-hour cap;
- Tesla T4, float32, exact validated environment;
- persistent output only under `/content/drive/MyDrive/latent-stroke-dynamics-phase-b0-recovery`;
- exact authorized bundle whose parent is the validated handoff commit;
- explicit one-time notebook switch after 145 tests and readiness checks.

## Not authorized

- any local recovery execution;
- a second cloud execution;
- automatic resume after interruption;
- tuning or rerunning against the result;
- formal Phase B0;
- Phase B1 saliency scheduling;
- Phase B2 RGB/high-resolution rendering.

The authorization begins unconsumed with zero completed executions. Starting the recovery is the one authorized attempt. If interrupted, preserve the Drive `.incomplete` directory and console log and stop for an audit.
