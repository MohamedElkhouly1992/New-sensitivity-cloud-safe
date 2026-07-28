"""Minimal check for the deadband fallback and monthly seasonal correction refinements."""
from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model

bldg = BuildingSpec(conditioned_area_m2=17948.4, floors=5, n_spaces=312)
cfg = HVACConfig(
    years=1,
    TIME_STEP_HOURS=24.0,
    APPLY_DUAL_SETPOINT_MIXED_MODE_TO_CORE=True,
    APPLY_PREDICTED_ZONE_DEADBAND_TO_CORE=True,
    APPLY_DEADBAND_FALLBACK_FIX_TO_CORE=True,
    APPLY_MONTHLY_COMPONENT_SEASONAL_CORRECTION=True,
    SEASONAL_FACTOR_DAMPING=0.65,
)

result = run_scenario_model(
    output_dir="example_seasonal_fallback_refinement_output",
    axis_mode="baseline_scenario",
    bldg=bldg,
    cfg=cfg,
    fixed_strategy="S2",
    fixed_severity="Mild",
    fixed_climate="C0_Baseline",
    include_baseline_layer=True,
    include_baseline_as_scenario=False,
)
print(result)
