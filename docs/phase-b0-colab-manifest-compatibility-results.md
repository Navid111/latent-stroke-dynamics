# Phase B0 Colab manifest compatibility — final result

## Outcome

The pinned Linux/Colab compatibility gate completed successfully as software but failed its exact data-continuity criterion.

```text
boundary tests: 8 passed in 12.18 seconds
manifest generation: completed in 66.62609805500006 seconds
hash gate passed: false
model resources loaded: false
scientific models trained: false
recovery output created: false
Google Drive accessed: false
historical incomplete directories touched: false
recovery authorized: false
```

## Environment

```text
platform: Linux-6.6.122+-x86_64-with-glibc2.35
machine: x86_64
Python: 3.13.15
NumPy: 2.5.2
Pillow: 12.3.0
PyTorch: 2.11.0+cpu
generation device: CPU
renderer boundary: PIL.ImageDraw.line
```

The standard-library `venv` setup initially failed because Colab's Python omitted `ensurepip`. This happened before tests or manifest generation. An ephemeral `pip --target` overlay was used instead, with the same exact pinned packages and unchanged repository source. This was an infrastructure-only amendment.

## Exact result

| Manifest | Frozen Mac SHA-256 | Pinned Linux SHA-256 | Match |
|---|---|---|---:|
| Diagnostic transitions | `d3101d29de97659a44932282fcbeed807405eecc1f678e71fd36e96a600d997a` | `97d7e6527b27ade5671732fd025e069cb4497c85e64ad6c853c0a3cf0cbfee0b` | no |
| Planner supervision | `02bef6101b0e380651301bbf7c8c0cf5e02c7c2a39e2dbab13e44fac1a9d186a` | `d5d0355c87f5f108b12a414c5e83c5a0bab733bcdfe41ddce9b5fc68e2feae62` | no |
| Train transitions | `7bb572b4d053649d22de75584615441b9d72c014f1a6128b435677e560c6304b` | `18551716942c747ee3daf8728bf1a8d1d21b9b075f85d71fe1365bcfd6a6e6e8` | no |
| Validation transitions | `234ff3b68399aea160ceb0665728d9f1d3d5971e1924ab36eef5c1537558c817` | `2b4fe2b782699538b91d3d13b453051fdb7e957d55fd371aba1cfdf56b44600a` | no |

The pinned Linux hashes and planner statistics are exactly the same as the first failed Colab recovery attempt. Therefore, dependency-version drift is not the cause. With the same NumPy, Pillow, PyTorch base version, seeds, source, counts, ordering, and CPU generation path, macOS ARM64 and Linux x86_64 still rasterize a sparse subset of non-no-op strokes differently.

## Conclusion

The evidence now supports same-version platform/architecture-dependent behavior at the `PIL.ImageDraw.line` boundary. Replacing frozen Mac hashes with Linux hashes would silently change the scientific data and is prohibited.

The version-pinning hypothesis is closed as a negative infrastructure result. No training occurred.

## Next path to training

The shortest safe route is a new immutable-data protocol:

1. Generate the exact frozen transition and planner payloads once on the Mac under the verified original environment.
2. Verify the four original manifest hashes before packaging.
3. Serialize the actual uint8 canvases, action rasters, no-op labels, planner tensors, progress labels, record metadata, and fitted progress statistics into an immutable portable bundle.
4. Hash every file and the complete archive.
5. In Colab, load and validate that bundle without invoking Pillow or regenerating any stroke.
6. Only after exact data validation, freeze a new recovery execution protocol and require a new one-time authorization.

This preserves the original scientific data while allowing GPU training on Colab. A platform-independent renderer can be studied later, but it is not the shortest route to the bachelor's-thesis deadline.
