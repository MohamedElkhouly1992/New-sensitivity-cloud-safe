# HVAC v3 soft-deadband, monthly availability, minimum-load and fan-schedule refinement

This deployment patch targets the remaining high daily CVRMSE/MAPE after the previous seasonal-fallback refinement. The latest error analysis showed that annual NMBE was acceptable, but February and October still collapsed into low-energy/deadband states while DesignBuilder continued to report significant HVAC operation.

## What was added

### 1. Soft thermostat/deadband activation

The previous predicted-zone gate was a hard switch:

```text
if T_zone_pred <= T_cool_set: Q_cool = 0
if T_zone_pred >= T_heat_set: Q_heat = 0
```

This can create artificial zero-load days. The new optional switch keeps the predicted-zone thermostat concept but replaces the hard threshold with a sigmoid gate:

```text
Q_cool = Q_cool_raw * G_cool
Q_heat = Q_heat_raw * G_heat
```

where `G_cool` and `G_heat` vary continuously between a small minimum fraction and 1.0. This prevents abrupt daily load collapse in shoulder seasons.

### 2. Monthly HVAC availability

The solver now has independent monthly availability arrays for cooling and heating. These are applied after the thermostat soft gate and before final component energy calculation. This helps represent the fact that DesignBuilder HVAC availability and operation schedules are seasonal and not determined only by outdoor-air temperature.

### 3. Monthly minimum operational load

A bounded monthly minimum load can be enforced for months where DesignBuilder shows persistent HVAC operation. This is especially important for February heating and October cooling, where the previous solver underpredicted because the gate reduced the load close to zero. The minimum is expressed as a fraction of design cooling/heating capacity and is still controlled by monthly availability and occupancy.

### 4. Schedule-based fan runtime

The fan is no longer allowed to behave only as a consequence of thermal load. A schedule-based fan runtime layer keeps fan operation linked to occupancy, HVAC activity, airflow setting, and monthly availability. This better reflects DesignBuilder air-loop and ventilation operation, where fan energy may exist even when the thermal load is small.

## New switches

```python
APPLY_SOFT_DEADBAND_ACTIVATION_TO_CORE = True
APPLY_MONTHLY_HVAC_AVAILABILITY_TO_CORE = True
APPLY_MONTHLY_MINIMUM_OPERATIONAL_LOAD_TO_CORE = True
APPLY_SCHEDULE_BASED_FAN_TO_CORE = True
```

## New important parameters

```python
SOFT_DEADBAND_SIGMOID_WIDTH_C = 0.75
SOFT_DEADBAND_MIN_GATE_FRACTION = 0.03
FAN_SCHEDULE_MIN_RUNTIME_FRACTION = 0.65
FAN_SCHEDULE_AIRFLOW_FLOOR_FRACTION = 0.75
```

The defaults are intentionally conservative. They are designed to reduce zero-load collapse without replacing the reduced-order physical solver with a direct monthly fit.

## New trace columns

The output CSV/Excel files include additional diagnostic columns:

```text
soft_deadband_activation_enabled
predicted_zone_cooling_soft_gate
predicted_zone_heating_soft_gate
monthly_hvac_availability_enabled
monthly_minimum_operational_load_enabled
monthly_cooling_availability
monthly_heating_availability
monthly_cooling_min_load_kw
monthly_heating_min_load_kw
monthly_min_cooling_added_kw
monthly_min_heating_added_kw
schedule_based_fan_enabled
P_fan_raw_before_schedule
P_fan_schedule_based
fan_runtime_fraction
fan_monthly_availability
fan_power_lift_kw
```

## Honest validation note

This bundle changes the operational timing mechanism that caused the February/October collapse, but it must still be validated against the DesignBuilder daily file. The target is not simply to improve annual total energy. The target is lower daily CVRMSE/MAPE and better monthly distribution while preserving acceptable NMBE.
