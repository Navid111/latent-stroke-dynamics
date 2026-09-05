# Interrupted quadratic-Bezier recovery validation results — 5 September 2026

## Result

The guarded recovery implementation at commit `46e0c6396f0425ed84812e8fbeef9ed675ef53e9` passed the dedicated no-output Colab validation.

- complete suite: **217 passed in 59.28 seconds**;
- notebook-measured pytest subprocess interval: **64.29310960399994 seconds**;
- environment: Python 3.13.15, Linux, NumPy 2.1.3, Pillow 11.3.0, Matplotlib 3.10.0;
- frozen runner commit: `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b`;
- seven frozen runner dependency paths unchanged;
- target-freeze SHA-256: `d7606da143f9b64e145cac95759dd3b29ff4a8bbf5edcf7aecce4d068930b494`;
- target-set SHA-256: `26bada941bfd8f49f09333d70d397364e82f5ddbb6e1228324f24fb9d2b30bfd`;
- recovery-plan SHA-256: `7127b1390cfbf3acd7cf85019e60abfedc70191e70d0130144a12c722072d18f`.

The validation did not mount Google Drive, access or modify the interrupted output, open images, reveal comparative metrics, create recovery output, train a model, use a learned model, or change closed experiments.

The explicit unauthorized probe stopped at the missing recovery-authorization file before any completed or `.incomplete` output was created. This confirms that the new recovery path is fail-closed at the separate authorization boundary.

## Gate decision

Implementation validation passed. Recovery execution remained unauthorized during validation. The next permitted repository action is a separate one-time authorization for this exact stable interrupted state, recovery plan, validated implementation commit, and frozen scientific runner. That authorization must not permit a fresh comparison or any overwrite, tuning, early reveal, or training.
