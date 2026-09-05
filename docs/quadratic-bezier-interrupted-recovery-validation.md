# Interrupted quadratic-Bezier recovery — no-output validation

## Purpose

Validate the guarded recovery implementation before any access to the preserved Google Drive output. This is a distinct implementation gate, not recovery execution and not a scientific run.

Exact recovery implementation commit:

```txt
46e0c6396f0425ed84812e8fbeef9ed675ef53e9
```

Frozen scientific runner commit:

```txt
398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b
```

## Procedure

1. Open `notebooks/quadratic_bezier_interrupted_recovery_validation.ipynb` from the `quadratic-bezier-extension` branch.
2. Use a fresh CPU runtime. Do not mount Google Drive.
3. Run all six code cells in order.
4. Expect the complete suite to report `217 passed`.
5. The explicit unauthorized probe must fail before creating either a completed or `.incomplete` output directory.
6. Download and return:
   - `quadratic_bezier_recovery_validation_report.json`;
   - `quadratic_bezier_recovery_pytest.txt`;
   - `quadratic_bezier_recovery_unauthorized_probe.txt`.
7. Disconnect and delete the validation runtime after the files are safely downloaded.

Expected report status:

```txt
quadratic_bezier_recovery_implementation_valid_no_outputs_unauthorized
```

The report must state that Google Drive was not mounted, the interrupted output was neither accessed nor modified, comparative metrics and images were not revealed, no recovery output was created, and recovery remains unauthorized.

## Boundary after validation

A passing report is implementation evidence only. Archive and review all three files before creating a separate one-time recovery authorization. Do not reopen the original execution notebook, run Cell 5, mount Drive during this validation, or execute `--execute-recovery` manually.
