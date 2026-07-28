from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model

bldg = BuildingSpec(
    building_type="Educational / University building",
    location="New Mansoura",
    conditioned_area_m2=17984.4,
    floors=4,
    n_spaces=310,
    occupancy_density_p_m2=0.111,
    lighting_w_m2=7.5,
    equipment_w_m2=11.77,
    airflow_m3h_m2=4.0,
    cooling_intensity_w_m2=150.0,
    heating_intensity_w_m2=75.0,
)

cfg = HVACConfig(
    years=1,
    TIME_STEP_HOURS=24.0,
    APPLY_DUAL_SETPOINT_MIXED_MODE_TO_CORE=True,
    APPLY_PREDICTED_ZONE_DEADBAND_TO_CORE=True,
    APPLY_SOFT_DEADBAND_ACTIVATION_TO_CORE=True,
    APPLY_MONTHLY_HVAC_AVAILABILITY_TO_CORE=True,
    APPLY_MONTHLY_MINIMUM_OPERATIONAL_LOAD_TO_CORE=True,
    APPLY_SCHEDULE_BASED_FAN_TO_CORE=True,
    APPLY_MONTHLY_COMPONENT_SEASONAL_CORRECTION=True,
    APPLY_OPERATIONAL_STATE_LAYER_TO_CORE=True,
)

outputs = run_scenario_model(
    output_dir="example_hidden_operational_state_output",
    axis_mode="baseline_scenario",
    bldg=bldg,
    cfg=cfg,
    fixed_strategy="S0",
    fixed_severity="Moderate",
    fixed_climate="C0_Baseline",
    include_baseline_layer=True,
    random_state=42,
)
print(outputs)
