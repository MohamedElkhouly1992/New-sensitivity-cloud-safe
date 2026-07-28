from pathlib import Path

from hvac_v3_engine import BuildingSpec, HVACConfig, synthetic_weather_timeseries, simulate_combo
from integrated_sensitivity_suite import OfficialHVACRunner


def test_summary_only_matches_standard_solver(tmp_path: Path):
    bldg = BuildingSpec(conditioned_area_m2=200.0)
    cfg = HVACConfig(years=1, TIME_STEP_HOURS=24.0, APO_POP=2, APO_ITERS=1)
    weather = synthetic_weather_timeseries(24.0, random_state=7)

    _, _, standard = simulate_combo(
        "S3", "Moderate", "C0_Baseline", bldg, cfg, weather,
        random_state=7, summary_only=False,
    )
    daily, annual, compact = simulate_combo(
        "S3", "Moderate", "C0_Baseline", bldg, cfg, weather,
        random_state=7, summary_only=True,
    )

    assert daily.empty
    assert annual.empty
    for key in [
        "Total Energy MWh", "Total Thermal HVAC Energy MWh", "Total Fan Energy MWh",
        "Total Pump Energy MWh", "Total Auxiliary Energy MWh", "Total Cost USD",
        "Total CO2 tonne", "Mean COP", "Mean Degradation Index",
        "Mean Comfort Deviation C", "Mean Cooling Load kW", "Mean Heating Load kW",
        "Occupied Discomfort Days", "Filter Replacements count", "HX Cleanings count",
    ]:
        assert abs(float(standard[key]) - float(compact[key])) < 1e-9


def test_persistent_evaluation_cache_resumes(tmp_path: Path):
    bldg = BuildingSpec(conditioned_area_m2=100.0)
    cfg = HVACConfig(years=1, TIME_STEP_HOURS=24.0, APO_POP=2, APO_ITERS=1)
    params = {"COP_COOL_NOM": 4.5}

    first = OfficialHVACRunner(
        bldg, cfg, 1, "synthetic", None, None, None, 42,
        None, None, "physics", 24.0,
        disk_cache_dir=tmp_path, summary_only=True,
    )
    result_1 = first.run(params, "S0", "Moderate", "C0_Baseline", 42)
    assert first.cache_stats()["solver_evaluations"] == 1
    assert len(list(tmp_path.glob("*.json"))) == 1

    resumed = OfficialHVACRunner(
        bldg, cfg, 1, "synthetic", None, None, None, 42,
        None, None, "physics", 24.0,
        disk_cache_dir=tmp_path, summary_only=True,
    )
    result_2 = resumed.run(params, "S0", "Moderate", "C0_Baseline", 42)
    assert resumed.cache_stats()["solver_evaluations"] == 0
    assert resumed.cache_stats()["disk_cache_hits"] == 1
    assert result_1 == result_2
