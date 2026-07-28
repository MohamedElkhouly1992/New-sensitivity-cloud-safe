# HVAC ROM-Degradation Suite — Strong Coupled Publication Version

This deployable Streamlit package extends the Publication Plus + EMS/Optimization version with **optional fully coupled diagnostic modules**. In earlier versions, the heat-exchanger, part-load COP, latent-load, and zone-level tabs mainly worked as post-processing/diagnostic layers. In this version, these modules can also modify the official Scenario Modeling outputs when their switches are enabled.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

or:

```bash
python -m streamlit run streamlit_app.py
```

## Main entry files

- `streamlit_app.py` — full graphical interface
- `hvac_v3_engine.py` — numerical engine and coupled calculations
- `report_addons.py` — reporting, plotting, validation, diagnostics, optimization helpers
- `requirements.txt` — dependencies

## Strong coupled switches

Open **Parameter Switches** and enable one or more of:

1. **Apply part-load COP to core**
   - Applies the selected PLR curve to `COP_eff` inside the time-step simulation.
   - Affects thermal HVAC energy, total energy, CO₂, cost, and objective value.

2. **Apply latent cooling to core**
   - Calculates humidity-ratio-based latent cooling from outdoor weather, indoor RH target, ventilation fraction, and infiltration.
   - Adds latent cooling to `Q_cool_kw` before calculating compressor/thermal HVAC energy.

3. **Apply HX air ΔP to fan**
   - Replaces/supplements the simple fan pressure term with a heat-exchanger air-side pressure-drop model:
     `ΔP_air = ΔP_clean × AF² × (1 + fouling_factor × degradation)`.
   - Affects fan power and total energy.

4. **Apply HX water ΔP to pump**
   - Replaces simple pump W/m² power with detailed water-side pressure-drop pump power:
     `P_pump = Vdot_water × ΔP_water / pump_efficiency`.
   - Affects pump energy and total energy.

5. **Apply HX UA degradation to capacity**
   - Reduces available cooling/heating capacity using:
     `capacity_factor = max(0.40, 1 − UA_loss_factor × degradation)`.
   - Can increase `capacity_unmet_kw` and comfort deviation.

6. **Use native zone-by-zone load sum**
   - If a zone table is supplied, loads are calculated per zone using each zone area, occupancy density, and schedule factors, then summed to building-level load.
   - Affects `Q_cool_kw`, `Q_heat_kw`, energy, CO₂, and comfort.

All coupled switches default to **OFF** to preserve the earlier core-model results. Turn them ON for the stronger publication formulation.

## New coupled output columns

The official time-step dataset can now include:

- `COP_base_before_PLR`
- `PLR`
- `PLR_modifier`
- `latent_cooling_kw`
- `dP_fan_Pa`
- `dP_water_kPa`
- `water_flow_m3h`
- `hx_capacity_factor`
- `capacity_unmet_kw`
- `zone_load_mode`
- `coupled_modules_active`

Energy-balance columns remain:

- `thermal_hvac_kwh_period`
- `fan_kwh_period`
- `pump_kwh_period`
- `auxiliary_kwh_period`
- `energy_kwh_period`

where:

```text
energy_kwh_period = thermal_hvac_kwh_period + fan_kwh_period + pump_kwh_period + auxiliary_kwh_period
```

## Recommended publication workflow

1. Run the baseline with all coupled switches OFF.
2. Enable one coupled switch at a time and compare outputs.
3. Enable the final set of coupled switches for the publication scenario.
4. Use **Sensitivity & Robustness** to quantify uncertainty.
5. Use **Advanced Plot Studio** to create final figures.
6. Use **Model Validation** when measured, EnergyPlus, DesignBuilder, BMS, or utility data are available.

## Honest limitation

This is still a reduced-order HVAC model. The coupled modules improve physical consistency and make diagnostics affect the official energy results, but they do not turn the model into a full heat-balance simulator like EnergyPlus. Case-specific validation is still required before making strong absolute-performance claims.

## Advanced HVAC Control Library update

This bundle adds a new **Advanced HVAC Control Library** tab. It includes deployable control templates for:

- ASHRAE Guideline 36-style high-performance control sequence approximation
- Demand-controlled ventilation through occupancy/airflow reset
- Fault-adaptive control recommendations
- Degradation-aware maintenance optimization recommendations
- Carbon-aware / price-aware control
- Peak demand limiting
- Zone priority control
- Hybrid degradation-aware EMS controller

The tab can score and rank control templates using energy, comfort, degradation, carbon, and fault-risk weights. A selected control template can be applied to the EMS settings and used in the main Scenario Modeling tab.

### Experimental controllers

The same tab includes two experimental research structures:

- **MPC experimental template:** prepares forecast and decision columns for future model predictive control work. It is not a validated live MPC solver.
- **RL experimental dataset specification:** defines state/action/reward fields for offline reinforcement learning dataset generation. It does not train or deploy an RL control policy.

These experimental tabs are intentionally separated from deployable control templates to avoid claiming real-time autonomous control without validation.

## Core KPI Impact Dashboard

This version adds a **Core KPI Impact Dashboard** tab. The tab is designed to answer whether each module is really connected to the core solver.

It runs paired simulations through `run_scenario_model()`:

1. A baseline engine run.
2. A modified engine run after applying the selected control/module settings.

It then compares official summary KPIs:

- Total Energy MWh
- Thermal HVAC MWh
- Fan MWh
- Pump MWh
- Auxiliary MWh
- Total CO2 tonne
- Total Cost USD
- Mean Comfort Deviation C
- Mean Degradation Index
- Mean COP

Supported impact cases:

- All-current setup vs original core
- Selected Advanced HVAC Control Library template
- Current EMS and operation schedule
- Coupled diagnostic modules
- Current parameter switches
- Each coupled module separately

The dashboard writes:

- `core_kpi_impact_all_cases.csv`
- one detailed CSV for each impact case

This tab calls the core engine directly. It does not rely on post-processing estimates.

## Live Core Solver Lab

This version adds a **Live Core Solver Lab** tab. It is designed for PhD defence demonstration and publication-grade time-resolved tracing.

Unlike a separate JavaScript or simplified demo model, this tab builds its live dataset by calling the same HVAC core solver functions used by the Scenario Modeling tab:

- weather loading and resampling through `_load_base_weather()`
- zone aggregation through `aggregate_zone_occupancy()`
- scenario solution through `simulate_combo()`

The live tab therefore includes the official effects of:

- EMS control
- operation scheduling
- S0/S1/S2/S3 strategy logic
- degradation and maintenance
- fan, pump, and auxiliary energy
- latent cooling coupling
- part-load COP coupling
- HX air-side pressure drop coupling
- HX water-side pressure drop coupling
- HX UA/capacity coupling
- native zone load coupling
- weather upload or synthetic weather

### Workflow

1. Open **Live Core Solver Lab**.
2. Select live years, strategy, severity, climate, and weather source.
3. Click **Build official live dataset using core solver**.
4. Click **Start** to animate the official solver rows.
5. Use **Step +1**, **Step +100**, or **Jump to end** for manual navigation.
6. Download `live_core_solver_timeseries.csv` for publication or supervisor review.

### Output files

The live tab writes:

- `live_core_solver_timeseries.csv`
- `live_core_solver_annual.csv`
- `live_core_solver_summary.csv`
- `live_core_weather_timeseries.csv`
- `live_core_metadata.json`

The playback is a live visualization of these official solver rows. For large hourly multi-year runs, the full CSV is saved to disk while browser playback can be limited to the first selected number of rows for stability.

## Live schematic layer update

This version adds a P&ID-style **Live HVAC schematic layer** inside the **Live Core Solver Lab** tab. The schematic is generated from the official core solver row currently being played back. It visualizes the same time-step variables used by the official solver, including ambient weather, AHU/fan pressure drop, chiller COP, HVAC load, pump energy, auxiliary energy, degradation index, comfort deviation, cumulative energy, CO2, and maintenance events.

The schematic is not a separate JavaScript/React solver. It is an SVG/HTML visualization layer driven by `live_core_solver_timeseries.csv`, which is produced by the same HVAC engine used by Scenario Modeling and Core KPI Impact.


## Export upgrade: tab-level and all-run result collection

This version adds export controls to every Streamlit tab. Each tab now includes an **Export this tab data/results** expander that can build:

- an Excel workbook containing the current setup, scalar session settings, available tab/session tables, and a result-file index;
- a ZIP package containing CSV versions of the exported tables and detected run files/charts such as CSV, XLSX, PDF, PNG, SVG, HTML, JSON and TXT outputs.

A final tab, **Run Result Collector**, collects the active setup plus all detected result folders and chart/report files into one downloadable package for each run. Use this tab after Scenario Modeling, Core KPI Impact, Live Core Solver, Sensitivity/Robustness, or diagnostic tabs have generated their outputs.

Recommended workflow:

1. Configure the building, HVAC, switches, EMS, schedule and controls.
2. Run Scenario Modeling or any analysis tab.
3. Use the export expander inside that tab to download its specific workbook/ZIP.
4. Open **Run Result Collector** to download a complete run archive for supervisor review or publication supplementary material.

## Thermal mass lag core-solver update

This patched deployment adds an optional **Apply thermal mass lag to core** switch. It is OFF by default so the earlier solver formulation remains backward-compatible. When enabled, the solver introduces one equivalent building-mass temperature state, `thermal_mass_temp_C`, which evolves through a first-order lag:

```text
T_mass,next = T_mass + beta * (T_eq - T_mass)
beta = 1 - exp(-dt_days / tau_days)
T_eq = T_out + solar_gain_equiv + internal_gain_equiv
```

The equivalent mass temperature then contributes delayed cooling or heating load:

```text
Q_mass,cool = w_cool * Q_cool_design * max(T_mass - T_sp, 0) / DT_REF_COOL
Q_mass,heat = w_heat * Q_heat_design * max(T_sp - T_mass, 0) / DT_REF_HEAT
```

### Why this was added

The original reduced-order solver mainly responds to current weather and current operation. DesignBuilder/EnergyPlus stores heat in walls, roofs, floors, internal partitions and zone mass. This produces delayed and smoothed daily HVAC loads. The new equivalent-mass state gives the reduced-order model limited thermal memory without changing the rest of the HVAC, degradation, APO, EMS, fan, pump or reporting structure.

### New configuration fields

```python
APPLY_THERMAL_MASS_LAG_TO_CORE=True
THERMAL_MASS_TIME_CONSTANT_DAYS=2.5
THERMAL_MASS_COOLING_WEIGHT=0.22
THERMAL_MASS_HEATING_WEIGHT=0.18
THERMAL_MASS_SOLAR_GAIN_C=2.0
THERMAL_MASS_INTERNAL_GAIN_C=1.0
```

### New output columns

- `thermal_mass_enabled`
- `thermal_mass_temp_C`
- `thermal_mass_equilibrium_C`
- `thermal_mass_next_C`
- `thermal_mass_beta`
- `thermal_mass_cooling_kw`
- `thermal_mass_heating_kw`
- `thermal_mass_delta_to_setpoint_C`

### Recommended use

1. Run the baseline with thermal mass OFF to reproduce the previous solver.
2. Enable **Apply thermal mass lag to core**.
3. Compare daily error metrics against DesignBuilder.
4. Calibrate only `THERMAL_MASS_TIME_CONSTANT_DAYS`, `THERMAL_MASS_COOLING_WEIGHT`, and `THERMAL_MASS_HEATING_WEIGHT` first.

Suggested ranges:

```text
THERMAL_MASS_TIME_CONSTANT_DAYS: 1.5 to 5.0
THERMAL_MASS_COOLING_WEIGHT: 0.10 to 0.35
THERMAL_MASS_HEATING_WEIGHT: 0.08 to 0.30
```

This module is a reduced-order approximation. It does not turn the solver into a full multi-layer heat-balance model; it adds interpretable thermal memory to improve daily pattern alignment.

## Thermal Mass Enhancement v2 — Energy-Neutral Lag

This enhanced bundle includes a safer thermal-mass formulation designed for DesignBuilder daily-pattern calibration.

The recommended setting is:

```python
APPLY_THERMAL_MASS_LAG_TO_CORE=True
THERMAL_MASS_MODE="EnergyNeutral"
THERMAL_MASS_TIME_CONSTANT_DAYS=2.5
THERMAL_MASS_LAG_STRENGTH=0.25
THERMAL_MASS_MAX_LOAD_SHIFT_FRACTION=0.20
```

`EnergyNeutral` mode delays/smooths the existing raw load. It does **not** add a large independent thermal-mass load. This was added because the first additive formulation could increase annual energy by about 20% by double-counting heat already represented by envelope, solar, infiltration, and internal-gain terms.

Use `THERMAL_MASS_MODE="Additive"` only for sensitivity testing.

Recommended validation workflow:

1. Run baseline with thermal mass OFF.
2. Run baseline with `EnergyNeutral` thermal mass ON.
3. Compare total energy, NMBE, daily MAPE, daily CVRMSE, monthly bias.
4. Tune only `THERMAL_MASS_TIME_CONSTANT_DAYS`, `THERMAL_MASS_LAG_STRENGTH`, and `THERMAL_MASS_MAX_LOAD_SHIFT_FRACTION`.
5. Accept the module only if daily MAPE/CVRMSE improve while total energy drift remains small.

## Dual-setpoint mixed heating/cooling mode enhancement

This bundle includes an optional DesignBuilder-style mixed-mode calculation for the core solver.

Enable it in **Parameter Switches**:

```text
Apply dual-setpoint mixed heating/cooling to core
```

or in code:

```python
cfg = HVACConfig(
    APPLY_DUAL_SETPOINT_MIXED_MODE_TO_CORE=True,
    MIXED_MODE_MIN_LOAD_FRACTION=0.01,
)
```

When enabled, the solver does not force the whole building into only one dominant heating or cooling mode. It calculates:

```text
P_HVAC = Q_cool / COP_cool + Q_heat / COP_heat
```

and reports separate cooling/heating thermal HVAC energy columns. This better approximates DesignBuilder/EnergyPlus multi-zone aggregation, where cooling and heating can occur in different zones or different hours of the same day.

Use `run_example_mixed_mode.py` for a quick test.


## Final refinement: predicted-zone deadband and DesignBuilder zone-file import

This version adds an optional predicted-zone-temperature activation gate for mixed heating/cooling mode and a DesignBuilder zone/grid file importer in the zone-specific occupancy workflow. See `PREDICTED_ZONE_DEADBAND_AND_DB_ZONE_IMPORT_REPORT.md`.

## Added refinement: deadband fallback fix + monthly component seasonal correction

This version includes two additional core-solver refinements designed to address the seasonal mismatch observed between the reduced-order model and the DesignBuilder reference.

### Constant deadband/fallback energy fix

When the predicted-zone deadband gate suppresses both heating and cooling, the solver no longer collapses into a repeated constant fan/pump/auxiliary-only value. A small bounded part-load/cycling load is recovered from the suppressed physical load using occupancy, solar intensity, and proximity to the heating/cooling setpoint boundary.

Relevant switches:

```python
APPLY_DEADBAND_FALLBACK_FIX_TO_CORE = True
DEADBAND_FALLBACK_SUPPRESSED_LOAD_FRACTION = 0.10
DEADBAND_FALLBACK_MAX_DESIGN_FRACTION = 0.08
```

### Monthly component seasonal correction

The monthly correction is applied at component level before final energy is written:

```python
APPLY_MONTHLY_COMPONENT_SEASONAL_CORRECTION = True
SEASONAL_FACTOR_DAMPING = 0.65
SEASONAL_FACTOR_MIN = 0.35
SEASONAL_FACTOR_MAX = 2.50
```

The solver writes traceable columns for the effective cooling, heating, fan, pump, and auxiliary factors. The included file `designbuilder_monthly_component_factors.csv` documents the default monthly factors derived from the DesignBuilder/model monthly mismatch.


## Added in this final operational-timing refinement

This deployment adds the next requested core-solver refinement after the seasonal/fallback version:

1. **Soft predicted-zone deadband activation** instead of hard zero-load gating.
2. **Monthly HVAC availability** for cooling and heating branches.
3. **Monthly minimum operational load** to prevent February/October load collapse when DesignBuilder shows active HVAC operation.
4. **Schedule-based fan runtime** so fan energy does not disappear only because thermal load is small.

Main switches:

```python
APPLY_SOFT_DEADBAND_ACTIVATION_TO_CORE=True
APPLY_MONTHLY_HVAC_AVAILABILITY_TO_CORE=True
APPLY_MONTHLY_MINIMUM_OPERATIONAL_LOAD_TO_CORE=True
APPLY_SCHEDULE_BASED_FAN_TO_CORE=True
```

New diagnostic columns are exported to CSV/Excel, including soft-gate factors, monthly availability, minimum-load additions, and fan schedule terms. See `SOFT_DEADBAND_AVAILABILITY_FAN_REFINEMENT_REPORT.md` and `monthly_operation_defaults.csv` for the scientific description and default monthly profiles.

## Hidden operational-state + genetic calibration refinement

This bundle includes an optional hidden operational-state layer (`APPLY_OPERATIONAL_STATE_LAYER_TO_CORE=True`). It classifies each timestep/day into interpretable operating regimes and applies bounded component-level factors before final energy is written. This targets residual seasonal mismatch after deadband, availability, minimum-load, and schedule-based fan refinements.

To calibrate the state factors from a DesignBuilder reference and model daily output, run:

```bash
python calibrate_operational_state_factors.py --designbuilder_csv "ALL DATA - Design builder Data.csv" --solver_csv baseline_no_degradation_daily.csv --out_csv operational_state_factors.csv
```

For defensible PhD/Q1 use, calibrate only on training years and report holdout or leave-one-year-out validation.

## Dynamic RC core solver refinement

This bundle includes an optional dynamic reduced-order core solver controlled by:

```python
APPLY_DYNAMIC_RC_CORE_SOLVER=True
```

When enabled, the load calculation carries equivalent zone-air and thermal-mass temperatures across timesteps. The model calculates a free-floating zone temperature and then estimates the cooling/heating needed to satisfy the dual setpoint band. This avoids a purely static relation between current weather and current load and helps represent thermal inertia and delayed building response.

Use hourly or sub-daily weather files for best results. New trace columns begin with `dynamic_`.

## Comprehensive Local and Global Sensitivity Studio

This integrated version adds a dedicated tab for signed OAT sensitivity, paired Monte Carlo robustness, Morris screening, Sobol first/total-order indices, benchmark cases, S3 ablation, optimizer robustness, and parameter-connectivity testing. All evaluations use the official HVAC v3 dynamic RC/degradation solver. Each table and figure can be downloaded separately, and each analysis also creates a complete ZIP. See `SENSITIVITY_STUDIO_GUIDE.md`.

## July 2026 Streamlit Cloud dependency fix

This package targets **Python 3.12** and includes the full CatBoost/SHAP surrogate workflow. NumPy is pinned to `1.26.4` because CatBoost 1.2.7 requires NumPy below 2.0. See `STREAMLIT_CLOUD_DEPENDENCY_FIX_PY312.md` for the corrected deployment procedure.

## Cloud-safe sensitivity resume

The Comprehensive Sensitivity Studio now uses a summary-only official solver path and per-evaluation JSON checkpoints. If a sensitivity calculation is interrupted, press the same Run button again to restore completed evaluations and continue with the missing cases. See `CALCULATION_INTERRUPTION_RESUME_FIX.md`.
