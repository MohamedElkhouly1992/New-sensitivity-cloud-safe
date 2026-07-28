# Thermal Mass Enhancement — Energy-Neutral Lag Version

## Why this enhancement was added

The first thermal-mass patch used an additive load formulation. It estimated an equivalent mass temperature and then added a delayed cooling or heating term to the raw HVAC load. That approach is physically interpretable, but in a reduced-order solver that already contains envelope, solar, infiltration and internal-gain terms, it can double-count heat that is already represented in the model. This explains why enabling thermal mass increased the total energy by about 20%.

This enhanced version changes the recommended thermal-mass behaviour from **additional load** to **energy-neutral lag smoothing**.

## New default method

The default mode is now:

```python
THERMAL_MASS_MODE = "EnergyNeutral"
```

In this mode, the solver delays and smooths the raw heating/cooling load rather than adding a new independent heat source. The correction is:

```text
Q_adjusted(t) = Q_raw(t) + clip[ s * (Q_lag(t) - Q_raw(t)), +/- cap ]
```

where:

- `Q_raw(t)` is the load already calculated by the existing solver,
- `Q_lag(t)` is the previous thermal-load memory,
- `s` is `THERMAL_MASS_LAG_STRENGTH`,
- `cap` is limited by `THERMAL_MASS_MAX_LOAD_SHIFT_FRACTION`.

The lag memory is updated after the accepted timestep, not during APO candidate evaluation. This protects the optimizer from mutating the thermal-memory state during candidate testing.

## Key parameters

```python
APPLY_THERMAL_MASS_LAG_TO_CORE = True
THERMAL_MASS_MODE = "EnergyNeutral"
THERMAL_MASS_TIME_CONSTANT_DAYS = 2.5
THERMAL_MASS_LAG_STRENGTH = 0.25
THERMAL_MASS_MAX_LOAD_SHIFT_FRACTION = 0.20
```

Legacy additive mode is still available:

```python
THERMAL_MASS_MODE = "Additive"
```

but it should be used only for sensitivity tests because it can increase annual energy.

## Expected behaviour

The enhanced module should primarily affect:

- daily load timing,
- peak smoothing,
- daily CVRMSE,
- daily MAPE,
- monthly transition-season shape.

It should not automatically increase annual energy by 20%. In the included example run, energy-neutral thermal mass changed total energy by about -0.1% compared with thermal mass disabled. This is the desired behaviour: the module redistributes load timing rather than creating a new annual heat source.

## New output columns

The solver now exports additional diagnostic columns:

- `thermal_mass_mode`
- `thermal_mass_base_cooling_kw`
- `thermal_mass_base_heating_kw`
- `thermal_mass_cooling_shift_kw`
- `thermal_mass_heating_shift_kw`
- `thermal_mass_adjusted_cooling_kw`
- `thermal_mass_adjusted_heating_kw`
- `thermal_mass_lag_cooling_memory_kw`
- `thermal_mass_lag_heating_memory_kw`
- `thermal_mass_load_conservation_note`

## Scientific defence statement

The enhanced thermal-mass module was reformulated as an energy-neutral delay mechanism. Instead of adding a new load term to the existing reduced-order thermal balance, the module now redistributes the raw cooling and heating loads through a bounded first-order lag. This better represents the role of building thermal mass as a temporal storage mechanism and reduces the risk of double-counting solar, internal, infiltration and envelope contributions already included in the solver. Therefore, the enhancement is more appropriate for reducing daily load-pattern mismatch against DesignBuilder while preserving annual energy balance.
