# Scripts

- `run_selector_component_confirmation.sh`: guarded launcher for one preregistered
  confirmation seed.
- `restore_validation_secure.py`: reconstructs the deterministic hidden holdout only
  after verifying the persisted visible split and manifest.
- `extract_trajectory.py`: converts one OpenMLE-Evo trajectory to a reviewable table.
- `counterfactual_parent_probe.py`: historical isolated forced-Improve probe. This
  is retained for transparency; its archive-wide support does not identify legal
  selector regret.

Scripts fail rather than overwrite existing confirmation outputs.

Runtime paths and local endpoints can be overridden with the `FRONTIS_*`
environment variables documented in each script. The historical counterfactual
probe requires `SANDBOX_CPU_API_KEY` to be supplied by the operator; no credential
is stored in this repository.
