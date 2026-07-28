from pathlib import Path
import pandas as pd

from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model


def main():
    out = Path('example_strong_coupled_output')
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
    print('Example completed.')
    print(result)
    df = pd.read_csv(result['dataset_csv'])
    print(df.head())


if __name__ == '__main__':
    main()
