from pathlib import Path
import pandas as pd
from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model

def main():
    out = Path("example_predicted_deadband_zone_run")
    zone_df = pd.read_csv("parsed_designbuilder_zone_table_from_uploaded_files.csv")
    bldg = BuildingSpec(
        building_type="Educational / University building",
        location="New Mansoura / DesignBuilder zone import",
        conditioned_area_m2=float(zone_df["area_m2"].sum()),
        floors=5,
        n_spaces=len(zone_df),
        occupancy_density_p_m2=float((zone_df["area_m2"]*zone_df["occ_density"]).sum()/zone_df["area_m2"].sum()),
        lighting_w_m2=float((zone_df["area_m2"]*zone_df["lighting_w_m2"]).sum()/zone_df["area_m2"].sum()),
        equipment_w_m2=float((zone_df["area_m2"]*zone_df["equipment_w_m2"]).sum()/zone_df["area_m2"].sum()),
        airflow_m3h_m2=4.0,
        cooling_intensity_w_m2=100.0,
        heating_intensity_w_m2=55.0,
    )
    cfg = HVACConfig(
        years=1,
        hvac_system_type="Chiller_AHU",
        TIME_STEP_HOURS=24.0,
        APPLY_DUAL_SETPOINT_MIXED_MODE_TO_CORE=True,
        APPLY_PREDICTED_ZONE_DEADBAND_TO_CORE=True,
        APPLY_NATIVE_ZONE_LOADS=True,
        HEATING_SETPOINT_C=22.0,
        COOLING_SETPOINT_C=24.0,
        PREDICTED_ZONE_DEADBAND_C=0.25,
        APO_POP=4,
        APO_ITERS=1,
    )
    result = run_scenario_model(
        output_dir=out,
        axis_mode="baseline_scenario",
        bldg=bldg,
        cfg=cfg,
        fixed_strategy="S2",
        fixed_severity="Moderate",
        fixed_climate="C0_Baseline",
        zone_df=zone_df,
        include_baseline_layer=True,
        random_state=42,
    )
    print("Completed:", result)
    print(pd.read_csv(result["dataset_csv"]).head())

if __name__ == "__main__":
    main()
