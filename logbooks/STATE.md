# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Ranking-aware formal-runner implementation  
**Status:** Development adjudicated and selected setting frozen; formal data unauthorized

## Validated development outcome

- tests: 69 passed;
- written-protocol integrity: passed;
- ranking weight: `1.0`;
- temperature: `0.05`;
- validation retrieval: 70.83%;
- diagnostic retrieval: 76.04%;
- diagnostic gain over matched MSE: 48.96 points;
- formal data generated: false.

The raw structural-NaN reporting failure remains archived unchanged. Method-applicable history values were all finite.

## Frozen formal inputs

- task-autoencoder state SHA-256: `95de3ecef8eeb7a350e862fa21185a168d9304870cb0c8391cbd008e88d93900`;
- latent-statistics SHA-256: `c2a3d781dab19a4714189d580dafb5ea95231af06021d3980beb495a3b85d903`;
- selected lambda/temperature: `1.0` / `0.05`;
- model seeds: `11`, `22`, `33`;
- formal seeds: `20261104`–`20261110`.

## Next action

Implement formal validation-only mode, matched MSE/ranking training, primary decision, stress evaluation, atomic outputs, and tests. Ask Navid to validate before a separate formal authorization commit.

Do not generate formal data yet.
