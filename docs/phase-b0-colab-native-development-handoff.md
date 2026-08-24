# Phase B0 cloud-native development handoff

## Decision

The aborted Mac attempt remains historical and untouched. It produced zero completed variants, zero checkpoints, and no scientific decision. The consumed recovery authorization remains permanently locked.

Phase B0 now proceeds as a **new cloud-native development experiment**. Linux x86-64 / Google Colab is the canonical renderer and training environment. The four Linux manifest hashes were reproduced independently twice before any model training result was observed, so they are frozen prospectively for this new experiment. The historical Mac hashes are not modified or reinterpreted.

## One-pass execution

One fresh Tesla T4 runtime will:

1. restore the exact authorized source and six required comparator resources;
2. run the complete test suite;
3. mount the dedicated Drive output root;
4. generate the frozen Linux transition and planner data on CPU;
5. verify the four already-frozen Linux hashes;
6. train both Phase B0 variants on `cuda:0`;
7. run diagnostics and the preregistered long-horizon comparison;
8. atomically finalize all artifacts in Drive; and
9. download one small completion handoff.

There is no Mac-to-Colab manifest exchange and no immutable-Mac-data packaging step.

## Current boundary

This handoff commit is unauthorized. Run the complete local suite once. Only after it passes may a separate direct-child authorization commit be issued and the single execution bundle built.

Formal Phase B0, Phase B1, and Phase B2 remain unauthorized.
