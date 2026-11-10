# JEPA4R Cube and Relocate workflows

This checkout composes the already smoke-tested Cube and Adroit/RH56 baselines. Their original algorithms, data protocols and method-specific configurations remain in their existing modules. Guide-specific recovery and evidence interfaces are being validated separately; see `SMOKE_VERIFICATION.json` for this checkout's actual acceptance state.

```bash
python run.py --list
python run.py --suite cube --list
python run.py --suite relocate --list
```

The original unambiguous flags also select their workflow: `--cube-count` selects Cube; `--env` or `--assets` selects Relocate. `--suite` makes that choice explicit. Conflicting suite flags are rejected. Commands retain the existing plan-only default; actual execution requires `--execute`.

Use the separate pinned environments in `requirements/worldmodel.txt`, `requirements/policy.txt`, `requirements/wm.txt`, `requirements/adroit.txt`, `requirements/rh56.txt` and `requirements/mpc.txt`. Cube and Relocate world-model NumPy versions differ. Keep simulator/BC and MPC dependency overlays separate from world-model training, and verify exact versions, CUDA operations, imports and rendering before running.

The existing branch instructions and verification records are preserved byte-for-byte:

- [Cube baseline instructions](provenance/baselines/cube/README.md)
- [Native Adroit and simulated RH56 instructions](provenance/baselines/adroit/README.md)
- [Cube baseline source identity](provenance/baselines/cube/BASELINE.json)
- [Adroit baseline source identity](provenance/baselines/adroit/BASELINE.json)

Those historical smoke records validate their respective frozen source snapshots. Acceptance of this integrated tree requires its own version-bound records. Smoke execution does not establish a new formal success rate. Large data, checkpoints and runtime evidence belong outside the source checkout.
