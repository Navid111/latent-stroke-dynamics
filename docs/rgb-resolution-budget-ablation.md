# RGB resolution x stroke-budget ablation

This is a bounded, exploratory engineering follow-up to the immutable fixed RGB
result. It does not train or load a model, alter Phase B0, replace a target, or
rerun the archived 96x96/210-stroke baseline.

## Fixed design

| Condition | Planning size | Stage budgets | Total | Action |
| --- | ---: | ---: | ---: | --- |
| A | 96x96 | 40/70/100 | 210 | Verify and reuse archived baseline |
| B | 96x96 | 80/140/200 | 420 | Run once |
| C | 128x128 | 40/70/100 | 210 | Run once |
| D | 128x128 | 80/140/200 | 420 | Run once |

All other painter settings, the exact five-target hash, seed 73, the 64-candidate
pool, 80/20 proposal mixture, straight opaque line primitive, exact rendered RGB
MSE selector, and 512x512 replay remain fixed.

The cross-resolution primary metric is mean RGB MSE against a separately resized
common 512x512 target. A condition is quantitatively eligible only when it lowers
that mean by at least 10 percent and no target becomes more than 5 percent worse.
If multiple conditions pass, the least expensive condition within one percent of
the best passing mean is selected for visual review. Code never fabricates the
required qualitative judgment: a quantitatively eligible result remains pending
until every-target output is inspected.

## Lifecycle

The runner verifies:

- source target hashes and their ordered combined hash;
- the exact baseline aggregate-summary hash;
- every expected baseline artifact listed in its manifest;
- protocol constants and two deterministic monotonic synthetic smoke tests;
- no training, learned model, or Phase B0 decision change.

It writes to a new `.incomplete` root, refuses overwrite/resume, and renames only
after all three new conditions, comparisons, hashes, and the aggregate summary
complete. Interruption or failure preserves `.incomplete` and records
`failure.json`.

## Local validation

First update and switch to the implementation branch, install the project in its
existing environment, and run the full test suite. Then validate against the
preserved baseline directory:

```bash
pytest -q
python run_rgb_resolution_budget_ablation.py \
  --validate-only \
  --input-dir local_targets/rgb-coarse-to-fine \
  --baseline-dir outputs/rgb-coarse-to-fine-fixed-seed73
```

Expected status:

```text
rgb_resolution_budget_ablation_valid_no_outputs
```

Validation must report `output_side_effects: false`. Do not execute conditions
B/C/D until the tests and validation report have been reviewed.

## Eventual one-time execution

```bash
python run_rgb_resolution_budget_ablation.py \
  --execute \
  --input-dir /path/to/exact/five/targets \
  --baseline-dir /path/to/rgb-coarse-to-fine-fixed-seed73 \
  --output-dir /new/path/rgb-resolution-budget-ablation-seed73
```

The output root must not exist. The web-sourced targets and all generated binary
images remain private and must not be committed without a rights basis. After
this ablation, experimental development freezes unless a separate defense-only
extension is explicitly approved after the thesis is submission-ready.
