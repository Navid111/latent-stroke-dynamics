# Stage 3 demonstration checkpoint review — 2026-08-22

**Status:** Success  
**Role:** Local deployment checkpoint for the learned painter  
**Formal paired-control result:** Unchanged

## Validation

- full repository suite: `33 passed in 2.12s`;
- checkpoint type: `stage3_pixel_mlp_demo`;
- architecture: `11 -> 64 -> 1` pixel MLP;
- parameters: 833;
- training rows: 1,000, seed `20260824`;
- validation rows: 200, seed `20260825`;
- train/validation overlap: zero;
- paired test rows generated: zero;
- paired test rows used for selection: false;
- reloaded predictions exactly matched in-memory predictions;
- state-dict SHA-256: `e32f3612f7a184e4e9b58f95a987551bd25cdb17ff1bf2b6be40fcf5781ea472`.

## Training behavior

Training completed all 30 epochs in 48.95 seconds on CPU. Validation balanced MSE decreased from `0.054394` at epoch 1 to its minimum of `0.000164624` at epoch 29. It rose slightly to `0.000175032` at epoch 30, while training loss continued to decrease.

The epoch-29 state was correctly restored and saved. The one-epoch validation increase is mild late-training overfit, not instability, and validates the decision to select the checkpoint using validation rather than the final epoch.

## Decision

The checkpoint is integrity-valid, data-clean, reproducibly identified, and ready for the development-only random/exact/learned painter smoke. The weight file remains local under ignored `checkpoints/` and must not be committed.
