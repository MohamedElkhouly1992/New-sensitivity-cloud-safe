from pathlib import Path
import pandas as pd

from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model


def run_case(name: str, cfg: HVACConfig):
    out = Path(f"compare_{name}")
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
    df = pd.read_csv(result["dataset_csv"])
    return {
        "case": name,
        "energy_MWh": float(df["energy_kwh_period"].sum() / 1000.0),
        "thermal_MWh": float(df["thermal_hvac_kwh_period"].sum() / 1000.0),
        "mean_mass_cool_shift_kw": float(df.get("thermal_mass_cooling_shift_kw", pd.Series([0.0])).mean()),
        "mean_mass_heat_shift_kw": float(df.get("thermal_mass_heating_shift_kw", pd.Series([0.0])).mean()),
        "output_csv": result["dataset_csv"],
    }


def main():
    base = dict(
        years=1,
        hvac_system_type="Chiller_AHU",
        TIME_STEP_HOURS=24.0,
        APO_POP=4,
        APO_ITERS=1,
        APPLY_PART_LOAD_COP_TO_CORE=True,
        APPLY_LATENT_LOAD_TO_CORE=True,
        APPLY_HX_AIR_PRESSURE_TO_FAN=True,
        APPLY_HX_WATER_PRESSURE_TO_PUMP=True,
        APPLY_HX_UA_TO_CAPACITY=True,
    )
    off = HVACConfig(**base, APPLY_THERMAL_MASS_LAG_TO_CORE=False)
    energy_neutral = HVACConfig(
        **base,
        APPLY_THERMAL_MASS_LAG_TO_CORE=True,
        THERMAL_MASS_MODE="EnergyNeutral",
        THERMAL_MASS_TIME_CONSTANT_DAYS=2.5,
        THERMAL_MASS_LAG_STRENGTH=0.25,
        THERMAL_MASS_MAX_LOAD_SHIFT_FRACTION=0.20,
    )
    rows = [run_case("thermal_mass_off", off), run_case("thermal_mass_energy_neutral", energy_neutral)]
    out = pd.DataFrame(rows)
    base_energy = out.loc[out["case"] == "thermal_mass_off", "energy_MWh"].iloc[0]
    out["energy_delta_pct_vs_off"] = 100.0 * (out["energy_MWh"] - base_energy) / max(abs(base_energy), 1e-9)
    out.to_csv("thermal_mass_mode_comparison.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
