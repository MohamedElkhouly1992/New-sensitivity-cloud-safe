from pathlib import Path
import pandas as pd
from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model

out = Path('example_thermal_mass_output')
bldg = BuildingSpec(
    building_type='Educational / University building',
    location='Example',
    conditioned_area_m2=1000.0,
    floors=2,
    n_spaces=10,
    occupancy_density_p_m2=0.08,
    lighting_w_m2=10.0,
    equipment_w_m2=8.0,
    airflow_m3h_m2=4.0,
    cooling_intensity_w_m2=100.0,
    heating_intensity_w_m2=55.0,
)
cfg = HVACConfig(
    years=1,
    hvac_system_type='Chiller_AHU',
    TIME_STEP_HOURS=24.0,
    APO_POP=4,
    APO_ITERS=1,
    APPLY_PART_LOAD_COP_TO_CORE=True,
    APPLY_LATENT_LOAD_TO_CORE=True,
    APPLY_HX_AIR_PRESSURE_TO_FAN=True,
    APPLY_HX_WATER_PRESSURE_TO_PUMP=True,
    APPLY_HX_UA_TO_CAPACITY=True,
    APPLY_THERMAL_MASS_LAG_TO_CORE=True,
    THERMAL_MASS_MODE='EnergyNeutral',
    THERMAL_MASS_TIME_CONSTANT_DAYS=2.5,
    THERMAL_MASS_LAG_STRENGTH=0.25,
    THERMAL_MASS_MAX_LOAD_SHIFT_FRACTION=0.20,
    THERMAL_MASS_COOLING_WEIGHT=0.06,
    THERMAL_MASS_HEATING_WEIGHT=0.05,
    THERMAL_MASS_SOLAR_GAIN_C=0.75,
    THERMAL_MASS_INTERNAL_GAIN_C=0.35,
)
result = run_scenario_model(
    output_dir=out,
    axis_mode='baseline_scenario',
    bldg=bldg,
    cfg=cfg,
    fixed_strategy='S2',
    fixed_severity='Moderate',
    fixed_climate='C0_Baseline',
    include_baseline_layer=True,
    random_state=42,
)
print('Thermal-mass example completed.')
print(result)
df = pd.read_csv(result['dataset_csv'])
cols = [c for c in ['day','T_amb_C','thermal_mass_mode','thermal_mass_temp_C','thermal_mass_next_C','thermal_mass_base_cooling_kw','thermal_mass_cooling_shift_kw','thermal_mass_adjusted_cooling_kw','thermal_mass_base_heating_kw','thermal_mass_heating_shift_kw','thermal_mass_adjusted_heating_kw','Q_cool_kw','Q_heat_kw','energy_kwh_period'] if c in df.columns]
print(df[cols].head(10))
