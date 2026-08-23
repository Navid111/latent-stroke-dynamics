# Phase B0 Google Colab CUDA preflight — passing result

## Evidential boundary

This is an infrastructure result, not a scientific model result. The preflight used deterministic random dummy tensors and loaded frozen historical resources only to verify raw-file and loaded-state hashes. It generated no renderer transitions, targets, state banks, candidate sets, scientific checkpoints, or scientific outputs. Phase B0 recovery, formal B0, B1, and B2 remained unauthorized.

## Exact source and bundle

- Source commit: `18afc1f3abf301f03417be73d817fea660ef6e45`
- Bundle: `dist/phase-b0-colab-preflight-18afc1f3abf3.tar.gz`
- Bundle SHA-256: `80ad2421ac888d239a97a88397402d2233efc5c3b9543f82250fbf0aad0aadbc`
- Bundle size: 2,198,467 bytes
- Frozen resources: 6
- Local suite: 120 passed in 8.39 seconds
- Colab suite: 120 passed

## Colab environment

- GPU: Tesla T4
- CUDA available: true
- Compute capability: 7.5
- GPU memory: 15,637,086,208 bytes
- Python: 3.13.15
- PyTorch: 2.11.0+cu128
- CUDA runtime: 12.8
- cuDNN: 91900

## Integrity checks

All six raw resource hashes matched the audited local files. Loaded task-autoencoder, three MSE-only latent-predictor, and pixel-predictor state hashes matched the frozen protocol. No ranking-aware model was loaded because the Phase B comparator uses only the MSE-only ensemble.

The deterministic CPU/CUDA output comparison passed. Maximum absolute error was `3.2782554626464844e-06`, below the preregistered `5e-4` absolute and relative tolerances.

Dummy optimizer losses were finite:

- joint prediction only: `0.1563444435596466`
- joint prediction + progress: `0.8266896605491638`

These dummy values are not scientific evidence.

## Throughput

- Median transition optimizer step: 37.69474150004726 ms
- Median planner optimizer step: 36.35918500003754 ms
- Maximum-epoch training estimate excluding validation: 7.984561109343001 minutes
- Three-times safety estimate excluding validation: 0.39922805546715 hours
- Peak CUDA allocation: 288,602,112 bytes

The timing supports implementing a CUDA recovery runner on free Colab. It does not estimate all CPU-bound rendering or long-horizon evaluation, so the six-hour total cap remains binding.

## Decision

Status: `phase_b0_colab_cuda_preflight_passed_recovery_unauthorized`.

The infrastructure is eligible for a guarded CUDA recovery implementation. This pass does not authorize renderer-data generation or training. The next gate is to implement and validate device-aware training, evaluation, exact resource loading, persistent artifact export, and interruption-safe lifecycle behavior while recovery remains unauthorized.
