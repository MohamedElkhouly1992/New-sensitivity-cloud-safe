# HVAC v3 Dynamic RC Core Solver Refinement

## Purpose

This refinement converts the core thermal-load calculation from a mostly static, instantaneous reduced-order solver into an optional dynamic reduced-order solver. The previous static formulation estimated the cooling/heating demand from the current timestep weather, solar gain, internal gain, occupancy, and setpoint. That structure can match annual energy but may miss day-to-day timing because it has weak thermal memory.

The dynamic refinement adds two state variables that are carried from one timestep to the next:

- `dynamic_zone_temp_C`: equivalent zone-air temperature
- `dynamic_mass_temp_C`: equivalent building thermal-mass temperature

The solver predicts a free-floating zone temperature before HVAC action and calculates the cooling/heating required to bring the dynamic zone state back toward the dual setpoint band.

## Scientific formulation

The simplified dynamic balance is:

```text
T_zone,free = T_zone + Δt / C_zone · [H_env(T_out - T_zone) + H_mz(T_mass - T_zone) + Q_solar,air + Q_internal,air]
```

Cooling is required when:

```text
T_zone,free > T_cool,set
```

Heating is required when:

```text
T_zone,free < T_heat,set
```

The equivalent mass node is updated using:

```text
T_mass,next = T_mass + Δt / C_mass · [H_mz(T_zone,next - T_mass) + Q_solar,mass + Q_internal,mass + H_env,mass(T_out - T_mass)]
```

To preserve compatibility with the existing reduced-order framework, the dynamic load is blended with the previous static load:

```text
Q_final = (1 - β)Q_static + βQ_dynamic
```

where `β = DYNAMIC_RC_LOAD_REPLACEMENT_FRACTION`.

## New configuration parameters

```python
APPLY_DYNAMIC_RC_CORE_SOLVER = True
DYNAMIC_RC_LOAD_REPLACEMENT_FRACTION = 0.75
DYNAMIC_ZONE_CAPACITANCE_KWH_M2K = 0.035
DYNAMIC_MASS_CAPACITANCE_KWH_M2K = 0.180
DYNAMIC_MASS_COUPLING_W_M2K = 2.50
DYNAMIC_ENV_UA_MULTIPLIER = 1.00
DYNAMIC_CONTROL_EFFECTIVENESS = 0.96
```

## New output columns

```text
dynamic_core_solver_enabled
dynamic_core_mode
dynamic_zone_temp_C
dynamic_mass_temp_C
dynamic_free_float_zone_temp_C
dynamic_zone_temp_next_C
dynamic_mass_temp_next_C
dynamic_cooling_required_kw
dynamic_heating_required_kw
dynamic_static_cooling_kw_before_dynamic
dynamic_static_heating_kw_before_dynamic
dynamic_static_blend_fraction
dynamic_env_heat_transfer_kw_per_K
dynamic_mass_coupling_kw_per_K
dynamic_zone_capacitance_kWh_per_K
dynamic_mass_capacitance_kWh_per_K
```

## Important use note

The dynamic solver is most meaningful when using hourly or sub-daily weather. It can still run with daily weather, but daily weather compresses the diurnal temperature/solar cycle and will reduce the benefit of the dynamic state model.

## Honest limitation

This is not a full EnergyPlus/DesignBuilder heat-balance solver. It is a reduced-order dynamic approximation. It improves temporal memory and thermal inertia but still requires calibration and validation against DesignBuilder or measured data.
