# Hidden Operational-State + Genetic Calibration Refinement

## Purpose

This refinement targets the residual seasonal mismatch observed after applying soft deadband activation, HVAC availability, minimum operational loads, and schedule-based fan runtime. The remaining error pattern was interpreted as an unknown operational-state mismatch: the DesignBuilder reference embeds calendar, academic, HVAC-availability, fan-runtime, warm-up, and low-operation behavior that is not always visible in exported daily energy files.

## What was added

A bounded hidden operational-state layer was added to the core solver. The solver classifies every timestep/day into one of seven interpretable regimes:

| Index | State |
|---:|---|
| 0 | R0_low_operation |
| 1 | R1_heating_dominated |
| 2 | R2_mixed_shoulder |
| 3 | R3_early_cooling |
| 4 | R4_peak_cooling |
| 5 | R5_late_cooling_academic |
| 6 | R6_extreme_high_load |

The state is inferred from month, outdoor temperature, occupancy, cooling/heating load shares, and weather/load severity indicators. The layer then applies bounded component factors to cooling, heating, fan, pump, and auxiliary energy before final energy is written.

## Why this helps

Monthly factors alone cannot distinguish between different operation types within the same month. For example, September may include both low-operation days and academic restart days. The hidden operational-state layer corrects by regime rather than by month alone.

## Genetic calibration script

The bundle includes:

```bash
python calibrate_operational_state_factors.py \
  --designbuilder_csv "ALL DATA - Design builder Data.csv" \
  --solver_csv baseline_no_degradation_daily.csv \
  --out_csv operational_state_factors.csv
```

The script tunes bounded state factors using a compact genetic algorithm. The objective function combines:

- daily CVRMSE,
- daily MAPE,
- mean absolute monthly NMBE,
- annual NMBE.

The recommended scientific workflow is to calibrate on training years only and validate on a holdout year or leave-one-year-out validation.

## New trace columns

The output includes traceable fields such as:

- `operational_state_layer_enabled`
- `operational_state_index`
- `operational_state_label`
- `operational_state_cooling_factor`
- `operational_state_heating_factor`
- `operational_state_fan_factor`
- `operational_state_pump_factor`
- `operational_state_aux_factor`
- `operational_state_cool_share`
- `operational_state_heat_share`

## Honest limitation

This layer is a physics-informed operational calibration, not a full reconstruction of DesignBuilder schedules. It improves the solver’s ability to represent hidden operating regimes, but it must be validated out-of-sample to avoid overfitting.
