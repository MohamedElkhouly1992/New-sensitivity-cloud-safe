# Streamlit width API migration fix

## Problem
Streamlit 1.59/1.60 emits repeated deprecation warnings for `use_container_width` and states that the parameter will be removed after 2025-12-31.

## Fix
All occurrences in the application were migrated as follows:

- `use_container_width=True` -> `width="stretch"`
- `use_container_width=False` -> `width="content"`

Files updated:

- `streamlit_app.py`: 47 calls
- `integrated_sensitivity_suite.py`: 3 calls

The change affects only UI sizing. It does not modify HVAC equations, dynamic RC calculations, degradation logic, EMS behavior, sensitivity sampling, Monte Carlo, Morris, Sobol, ablation, optimization, or exports.

## Deployment
Replace the repository contents with this package, commit and push, then reboot the Streamlit Community Cloud app.
