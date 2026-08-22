# Frozen full representation extension — authorized run handoff

## Validation record

Navid reported:

- `51 passed in 6.09s`;
- full-command status `full_command_valid`;
- autoencoder parameter count `49,569`;
- all four dynamics parameter counts matched the frozen values;
- output directory available;
- authorized run not started;
- primary/stress data not generated;
- development metrics changed no setting;
- historical decisions unchanged.

The single frozen extension run is now authorized.

## Exact command

After pulling the authorization record, run exactly:

```bash
git pull --ff-only
source .venv/bin/activate
python experiments/10_representation_extension_full.py --run-frozen-extension
```

Do not add flags, change thread counts, or run a second process. Keep the Mac connected to power, keep the terminal open, and prevent sleep. The full run generates 1,000/200/300 primary transitions, four 100-example stress slices, three autoencoder seeds, and twelve dynamics fits, so it may take substantially longer than the development smoke.

## Success artifact

On completion, send this actual file:

```text
outputs/representation-extension-2026-08-22/extension_summary.json
```

Do not rerun or retune after any completed outcome.

## Failure handling

If the process raises an error or is interrupted:

1. do not execute the command again;
2. preserve the complete traceback;
3. preserve `outputs/representation-extension-2026-08-22.incomplete/` exactly;
4. send the traceback before deleting, moving, or editing anything.

An implementation/infrastructure interruption can be reviewed without treating a partial run as scientific evidence. No silent restart is allowed.
