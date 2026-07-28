# Dual-Setpoint Mixed Heating/Cooling Mode Patch

This patch adds an optional DesignBuilder-style mixed heating/cooling mode to the reduced-order HVAC solver.

## Purpose

The previous reduced-order calculation selected one dominant branch per timestep:

```text
if Q_cool >= Q_heat: mode = cooling and Q_HVAC = Q_cool
else: mode = heating and Q_HVAC = Q_heat
```

This is computationally simple, but it does not reproduce the way a multi-zone DesignBuilder/EnergyPlus simulation can produce heating and cooling energy in the same day or timestep. In a large multi-zone building, some zones may need cooling because of solar/internal gains while other zones may still need heating because of envelope heat loss or schedule effects.

## New optional switch

```python
APPLY_DUAL_SETPOINT_MIXED_MODE_TO_CORE = True
MIXED_MODE_MIN_LOAD_FRACTION = 0.01
```

When enabled, the solver keeps both thermal branches:

```text
Q_HVAC = Q_cool + Q_heat
P_HVAC = Q_cool / COP_cool + Q_heat / COP_heat
```

The output mode becomes:

- `cooling` if only cooling is active;
- `heating` if only heating is active;
- `mixed` if both heating and cooling are active;
- `deadband` if neither branch exceeds the active-load threshold.

## New output columns

The official results dataset can now include:

```text
dominant_mode
mixed_mode_enabled
Q_cool_active
Q_heat_active
cooling_hvac_kwh_period
heating_hvac_kwh_period
COP_cooling_before_PLR
COP_heating_before_PLR
COP_cooling_with_PLR
COP_heating_with_PLR
PLR_cooling
PLR_heating
PLR_modifier_cooling
PLR_modifier_heating
```

## Scientific interpretation

This modification does not convert the solver into a full multi-zone heat-balance simulator. It is still a reduced-order approximation. However, it removes the unrealistic restriction that the entire building must be either heating or cooling during a timestep. This makes the solver closer to DesignBuilder aggregation behavior, especially for shoulder seasons and high-solar/internal-gain periods.

## Recommended validation workflow

1. Run the baseline with the mixed-mode switch OFF.
2. Run the same baseline with the mixed-mode switch ON.
3. Compare daily MAPE, CVRMSE, WMAPE, NMBE, monthly bias, and annual total error.
4. Keep the switch ON only if the daily pattern improves without introducing unacceptable total-energy bias.

The strongest defence statement is:

> A dual-setpoint mixed-mode approximation was added to the reduced-order model to better represent the multi-zone behavior of the DesignBuilder reference simulation, where heating and cooling demands can occur within the same day or timestep. The modification calculates cooling and heating loads independently and sums their electricity using separate cooling and heating COPs.
