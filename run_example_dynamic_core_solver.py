from pathlib import Path

from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model


if __name__ == "__main__":
    output_dir = Path("example_dynamic_core_solver_output")
    bldg = BuildingSpec(
        conditioned_area_m2=5000.0,
        floors=4,
        n_spaces=40,
        occupancy_density_p_m2=0.08,
        lighting_w_m2=10.0,
        equipment_w_m2=8.0,
        airflow_m3h_m2=4.0,
        cooling_intensity_w_m2=100.0,
        heating_intensity_w_m2=55.0,
    )
    cfg = HVACConfig(
        years=1,
        TIME_STEP_HOURS=24.0,
        APPLY_DYNAMIC_RC_CORE_SOLVER=True,
        DYNAMIC_RC_LOAD_REPLACEMENT_FRACTION=0.75,
        APPLY_DUAL_SETPOINT_MIXED_MODE_TO_CORE=True,
        APPLY_PREDICTED_ZONE_DEADBAND_TO_CORE=True,
        APPLY_SOFT_DEADBAND_ACTIVATION_TO_CORE=True,
        APPLY_MONTHLY_HVAC_AVAILABILITY_TO_CORE=True,
        APPLY_MONTHLY_MINIMUM_OPERATIONAL_LOAD_TO_CORE=True,
        APPLY_SCHEDULE_BASED_FAN_TO_CORE=True,
        APPLY_OPERATIONAL_STATE_LAYER_TO_CORE=True,
        APPLY_MONTHLY_COMPONENT_SEASONAL_CORRECTION=True,
    )
    paths = run_scenario_model(
        output_dir=output_dir,
        axis_mode="baseline_scenario",
        bldg=bldg,
        cfg=cfg,
        weather_mode="synthetic",
        fixed_strategy="S2",
        fixed_severity="Moderate",
        fixed_climate="C0_Baseline",
        degradation_model="physics",
        include_baseline_layer=True,
        time_step_hours=24.0,
    )
    print("Dynamic core solver example completed.")
    print(paths)
