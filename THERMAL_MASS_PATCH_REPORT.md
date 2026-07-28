# HVAC v3 Thermal Mass Lag Patch Report

## What was changed

Only an optional thermal-mass-lag layer was added to the existing bundle. The rest of the solver structure, degradation logic, APO search, EMS logic, component energy balance, reports, and Streamlit workflow were left intact.

## New physical state

The patch adds an equivalent building-mass temperature:

```text
T_mass
```

This state represents the average thermal memory of walls, roof, floors, internal partitions, furniture and other internal mass. It is not a full DesignBuilder wall-layer heat-balance calculation.

## State update equation

```text
T_mass,next = T_mass + beta * (T_eq - T_mass)
beta = 1 - exp(-dt_days / tau_days)
```

where:

```text
T_eq = T_out + solar_gain_equivalent + internal_gain_equivalent
```

## Load coupling

If the equivalent mass is warmer than the setpoint, it adds a delayed cooling load. If it is colder than the setpoint, it adds a delayed heating load.

```text
Q_mass,cool = w_cool * Q_cool_design * max(T_mass - T_sp, 0) / DT_REF_COOL
Q_mass,heat = w_heat * Q_heat_design * max(T_sp - T_mass, 0) / DT_REF_HEAT
```

## Why it should help daily errors

The previous reduced-order model was nearly instantaneous: it reacted mainly to current ambient temperature, humidity, solar radiation and occupancy. DesignBuilder has thermal memory because it dynamically stores and releases heat in construction layers and internal mass. This difference can shift the daily energy peak and raise CVRMSE even when the annual energy total is close. The patch adds limited thermal memory, so daily loads are smoother and less dependent on only the current day.

## What it does not do

It does not guarantee lower MAPE or CVRMSE for every dataset. The improvement depends on calibration of the time constant and weights against the DesignBuilder baseline. It should be treated as a physically interpretable refinement, not as final validation by itself.

## How to enable

In Streamlit, open **Parameter Switches** and enable:

```text
Apply thermal mass lag to core
```

Or in Python:

```python
cfg = HVACConfig(
    APPLY_THERMAL_MASS_LAG_TO_CORE=True,
    THERMAL_MASS_TIME_CONSTANT_DAYS=2.5,
    THERMAL_MASS_COOLING_WEIGHT=0.22,
    THERMAL_MASS_HEATING_WEIGHT=0.18,
)
```
