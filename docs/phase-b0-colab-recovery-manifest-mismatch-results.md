# Phase B0 Colab recovery — consumed manifest-mismatch attempt

## Outcome

The one-time Colab recovery authorization was consumed on 2026-08-24. The runner stopped at the fail-closed data-manifest continuity gate before any scientific training.

- Failure stage: `data_manifest_hash_verification`
- Training started: **no**
- Models/checkpoints/targets/histories/decision/run config created: **no**
- Final output created: **no**
- Cloud `.incomplete` evidence preserved: **yes**
- Rerun under the old authorization: **forbidden**
- Formal Phase B0, B1, and B2: **unauthorized**

The immutable issuance record remains in the repository as historical evidence. Runtime authorization code now treats that issuance as permanently consumed.

## Cross-environment comparison

The Mac originals were re-audited without modification and matched all four frozen expected hashes. The downloaded Colab manifests were compared record by record.

| Split | Samples | Matching | Differing | Fraction | No-op mismatches |
|---|---:|---:|---:|---:|---:|
| Train | 2,048 | 2,011 | 37 | 1.8066% | 0 |
| Validation | 512 | 505 | 7 | 1.3672% | 0 |
| Diagnostic | 512 | 507 | 5 | 0.9766% | 0 |

Across the three transition splits, 49 of 3,072 fingerprints differed (about 1.60%). Record counts, seeds, ordering, metadata, no-op transitions, and transition-manifest byte sizes matched. Matching fingerprints resumed after mismatch records.

Planner training and validation records also matched exactly. Only image-derived progress statistics differed:

- Local mean: `-0.00015118662849999964`
- Colab mean: `-0.00017171769286505878`
- Local standard deviation: `0.001871041371487081`
- Colab standard deviation: `0.0018368015298619866`

## Interpretation

The evidence strongly narrows the failure to sparse rendering differences for non-no-op strokes, most likely version- or platform-dependent behavior in `PIL.ImageDraw.line`. It does not support RNG-sequence drift, manifest corruption, or model behavior as the cause.

The exact NumPy and Pillow versions from the deleted Colab runtime were not captured. This does not weaken the fail-closed result or require recreating that runtime. The original cloud evidence is preserved on Drive, and the local environment was captured as NumPy `2.5.2` and Pillow `12.3.0`.

## Next permitted step

Only a new, manifest-only compatibility protocol may run next. It must:

1. pin NumPy `2.5.2` and Pillow `12.3.0`;
2. record the environment before generation;
3. generate only the four continuity manifests;
4. load no model resources and create no checkpoints, targets, histories, decisions, or recovery output;
5. require all four original hashes to match exactly;
6. leave both historical `.incomplete` directories untouched; and
7. grant no recovery or later-phase authorization.

A passing compatibility check is necessary but not sufficient for training. Any scientific recovery would require a separately frozen protocol, a fresh output root, successful validation, and a new one-time authorization.
