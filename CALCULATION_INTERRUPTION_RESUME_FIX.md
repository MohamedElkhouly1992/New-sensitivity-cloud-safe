# Calculation Interruption and Resume Fix

## Problem

Long sensitivity studies can be interrupted on Streamlit Community Cloud because each analysis may call the complete HVAC model hundreds or thousands of times. The former sensitivity runner requested the full, very wide timestep DataFrame from `simulate_combo()` for every model evaluation even though the sensitivity methods only require scalar summary KPIs. S3 cases are additionally expensive because each timestep executes `APO_POP × APO_ITERS` control candidates.

## Implemented corrections

### 1. Summary-only official solver mode

`hvac_v3_engine.simulate_combo()` now accepts:

```python
summary_only=True
```

In this mode, the same timestep equations, dynamic RC states, EMS actions, degradation, maintenance, component-energy calculations, comfort calculations, and S3 optimizer are executed. Only scalar running aggregates are retained; the full timestep table is not accumulated in RAM.

The standard solver remains unchanged because `summary_only=False` is still the default.

### 2. Persistent evaluation checkpoints

Every completed sensitivity model evaluation is saved as an atomic JSON checkpoint in:

```text
<selected output folder>/comprehensive_sensitivity/_evaluation_cache/
```

The cache signature includes the building, HVAC configuration, years, time step, weather values, zone table, operation schedule, degradation model, strategy, severity, climate, seed, analysis mode, and uncertain-parameter values.

If the app is interrupted, open the same sensitivity analysis and press the same **Run** button again. Previously completed evaluations are restored from the checkpoint cache and only missing evaluations are calculated.

### 3. Cloud-safe sensitivity time step

The sensitivity tab now has a separate **Sensitivity solver time step** selector. The default is Daily (24 h), which is recommended for cloud screening. The main scenario time step is not changed.

### 4. S3 optimizer workload profiles

The sensitivity tab provides:

- Current model settings
- Cloud screening: 8 × 4
- Fast diagnostic: 4 × 2

Reduced profiles are intended only for software checks and preliminary screening. Final manuscript results should use the validated optimizer settings.

### 5. Native thread limits

BLAS/OpenMP thread counts are limited to one on startup to avoid memory and CPU oversubscription on small cloud workers.

### 6. Streamlit file watcher disabled in production

`.streamlit/config.toml` now uses:

```toml
[server]
fileWatcherType = "none"
```

This prevents result and checkpoint file activity from causing unnecessary source-reload behavior.

## Verification

- Python syntax compilation: passed.
- Existing integrated sensitivity test: passed.
- Summary-only S0 result equivalence: passed to floating-point precision.
- Summary-only S3 result equivalence: passed to floating-point precision.
- Persistent checkpoint resume test: passed; the second runner restored the result without executing the solver.
- Total automated tests: 3 passed.

## Recommended cloud workflow

1. Use Daily sensitivity time step.
2. Use one analysis year for the diagnostic run.
3. Start with Morris 5–10 trajectories, Sobol N=16–32, or Monte Carlo 10–20 samples.
4. If interrupted, reopen the tab and press the same Run button again.
5. Increase the sample size in stages after confirming stability.
6. Run final high-resolution Sobol and large Monte Carlo studies on a local workstation or a dedicated server.

## Important limitation

Checkpoint files are local to the running deployment. They survive normal Streamlit reruns and allow recovery from an interrupted calculation while the deployment storage remains available. They are not a substitute for durable database/object storage and should not be assumed to survive repository redeployment or platform replacement of the application container.
