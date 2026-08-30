# Fixed RGB coarse-to-fine painter

This is a bounded qualitative engineering extension. It does not train or load
a learned model and does not alter any frozen Phase B0 decision.

The runner verifies all five fixed source hashes, preserves aspect ratio with
white padding, plans at 96x96 RGB, proposes 64 unique changing candidates per
pool, uses an 80/20 error-guided and uniform mixture, chooses exactly rendered
RGB outcomes by target-pixel MSE, executes only improvements, preserves the best
frame, uses global/structure/detail stages, and replays at 512x512.

Source images remain local in `local_targets/rgb-coarse-to-fine/`. Generated
outputs remain under `outputs/` and are not committed.

Validation command:

    python paint_rgb_coarse_to_fine.py --validate-only

Do not use `--execute` until the local tests and validation report have been
reviewed. The eventual complete run must use the frozen five-target manifest
and fixed configuration without post-result target replacement or tuning.
