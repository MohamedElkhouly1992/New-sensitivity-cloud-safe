# HVAC v3 Final Refinement — Predicted-Zone Deadband + DesignBuilder Zone Import

## What changed

This bundle keeps the existing dual-setpoint mixed mode, but adds a predicted-zone-temperature activation gate:

- Cooling is calculated only when the predicted zone temperature exceeds the cooling setpoint.
- Heating is calculated only when the predicted zone temperature falls below the heating setpoint.
- Mixed mode remains enabled when both heating and cooling are physically justified.

This prevents winter cooling from being activated only because solar/internal gains exist.

## New core switch

```python
APPLY_PREDICTED_ZONE_DEADBAND_TO_CORE = True
```

Recommended parameters:

```python
HEATING_SETPOINT_C = 22.0
COOLING_SETPOINT_C = 24.0
PREDICTED_ZONE_DEADBAND_C = 0.25
PREDICTED_ZONE_OUTDOOR_WEIGHT = 0.35
PREDICTED_ZONE_SOLAR_GAIN_C = 2.50
PREDICTED_ZONE_INTERNAL_GAIN_C = 1.50
PREDICTED_ZONE_THERMAL_MASS_WEIGHT = 0.25
```

## New DesignBuilder zone import

The Streamlit Scenario Modeling tab now allows:

`Use zone-specific occupancy input` -> `Upload DesignBuilder zone/grid files`

Supported DesignBuilder exports include:

- zones.csv
- internal loads.csv
- set points.csv
- zone lighting.csv
- zone hvac.csv
- zone Mechanical ventilation.csv
- zone Natural Ventilation.csv

The parser extracts zone area, occupancy density, lighting W/m², equipment W/m², heating/cooling setpoints, and ventilation fields when available.

## New output columns

The results include:

- `predicted_zone_deadband_enabled`
- `predicted_zone_temp_C`
- `predicted_zone_heating_setpoint_C`
- `predicted_zone_cooling_setpoint_C`
- `predicted_zone_cooling_gate`
- `predicted_zone_heating_gate`
- `predicted_zone_cooling_suppressed_kw`
- `predicted_zone_heating_suppressed_kw`

## Scientific interpretation

Before this refinement, the solver could convert solar and internal gains directly into cooling load even during winter. That can produce high cooling load in January even when the building should first use those gains to offset heating demand. The new predicted-zone gate makes the reduced-order model closer to DesignBuilder thermostat behavior: a cooling/heating load is accepted only if the estimated zone state crosses the scheduled cooling/heating setpoint.
