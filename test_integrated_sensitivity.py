from pathlib import Path

from hvac_v3_engine import BuildingSpec, HVACConfig
from integrated_sensitivity_suite import (
    OfficialHVACRunner,
    default_parameter_specs,
    run_local_oat,
)


def test_official_solver_local_oat(tmp_path: Path):
    building = BuildingSpec(conditioned_area_m2=250.0, n_spaces=3)
    cfg = HVACConfig(
        years=1,
        TIME_STEP_HOURS=24.0,
        APPLY_DYNAMIC_RC_CORE_SOLVER=False,
        APPLY_MONTHLY_COMPONENT_SEASONAL_CORRECTION=False,
        APPLY_OPERATIONAL_STATE_LAYER_TO_CORE=False,
        APPLY_MONTHLY_HVAC_AVAILABILITY_TO_CORE=False,
        APPLY_MONTHLY_MINIMUM_OPERATIONAL_LOAD_TO_CORE=False,
        APPLY_SCHEDULE_BASED_FAN_TO_CORE=False,
    )
    runner = OfficialHVACRunner(
        building, cfg, 1, "synthetic", None, None, None, 42,
        None, None, "physics", 24.0,
    )
    specs = default_parameter_specs(building, cfg)[:1]
    paths = run_local_oat(
        runner, specs, "S0", "Moderate", "C0_Baseline", 42, tmp_path
    )
    assert Path(paths["indices_csv"]).exists()
    assert Path(paths["zip"]).exists()
