from pathlib import Path
import pandas as pd

from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model


def main():
    out = Path("example_mixed_mode_output")
    bldg = BuildingSpec(
        building_type="Educational / University building",
        location="Example",
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
        hvac_system_type="Chiller_AHU",
        TIME_STEP_HOURS=24.0,
        APO_POP=4,
        APO_ITERS=1,
        APPLY_DUAL_SETPOINT_MIXED_MODE_TO_CORE=True,
        MIXED_MODE_MIN_LOAD_FRACTION=0.01,
        APPLY_THERMAL_MASS_LAG_TO_CORE=True,
        THERMAL_MASS_MODE="EnergyNeutral",
    )
    result = run_scenario_model(
        output_dir=out,
        axis_mode="baseline_scenario",
        bldg=bldg,
        cfg=cfg,
        fixed_strategy="S2",
        fixed_severity="Moderate",
        fixed_climate="C0_Baseline",
        include_baseline_layer=True,
        random_state=42,
    )
    print("Mixed-mode example completed.")
    print(result)
    df = pd.read_csv(result["dataset_csv"])
    cols = [
        "mode",
        "dominant_mode",
        "mixed_mode_enabled",
        "Q_cool_kw",
        "Q_heat_kw",
        "Q_HVAC_kw",
        "cooling_hvac_kwh_period",
        "heating_hvac_kwh_period",
        "thermal_hvac_kwh_period",
    ]
    print(df[[c for c in cols if c in df.columns]].head(10))


if __name__ == "__main__":
    main()
