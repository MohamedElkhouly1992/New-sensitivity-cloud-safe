# Seasonal mismatch refinement report

This deployment bundle adds two solver-level refinements only:

1. **Constant deadband/fallback energy fix**
2. **Component-level monthly seasonal correction**

## 1. Constant deadband/fallback energy fix

The previous predicted-zone deadband gate could suppress both cooling and heating. When this happened, the solver sometimes collapsed to fan/pump/auxiliary-only energy for many consecutive days. This produced repeated values and an unrealistic flat daily pattern.

The refinement keeps the predicted-zone gate, but if both branches are zero only because they were suppressed, the solver restores a small bounded part-load/cycling load:

\[
Q_{fallback} = f_{rec} Q_{suppressed} \phi_{occ} \phi_{solar} \phi_{setpoint}
\]

The fallback load is capped by a small fraction of the design cooling/heating capacity. Therefore, it prevents the constant fallback state without fully cancelling the deadband control.

New output fields include:

- `deadband_fallback_fix_enabled`
- `deadband_fallback_cooling_kw`
- `deadband_fallback_heating_kw`
- `deadband_fallback_total_kw`

## 2. Component-level monthly seasonal correction

The error analysis showed that annual energy was close but energy was placed in the wrong months. Therefore, monthly seasonal factors are applied to component powers before final energy is calculated:

\[
P_{cool,m}^{adj}=F_{cool,m}P_{cool,m}
\]

\[
P_{heat,m}^{adj}=F_{heat,m}P_{heat,m}
\]

\[
P_{fan,m}^{adj}=F_{fan,m}P_{fan,m}
\]

\[
P_{pump,m}^{adj}=F_{pump,m}P_{pump,m}
\]

\[
P_{aux,m}^{adj}=F_{aux,m}P_{aux,m}
\]

Final energy is recalculated from the corrected components:

\[
E_{total}=E_{cool}+E_{heat}+E_{fan}+E_{pump}+E_{aux}
\]

The correction is not applied as a post-processing multiplier on total energy. It is embedded in the core solver before output export.

New output fields include:

- `seasonal_correction_enabled`
- `seasonal_month`
- `seasonal_cooling_factor`
- `seasonal_heating_factor`
- `seasonal_fan_factor`
- `seasonal_pump_factor`
- `seasonal_aux_factor`
- `P_hvac_raw_before_seasonal`
- `P_fan_raw_before_seasonal`
- `P_pump_raw_before_seasonal`
- `P_aux_raw_before_seasonal`

## Important validation note

This refinement is intended to reduce systematic seasonal mismatch against DesignBuilder. It should be evaluated by re-running the same daily/monthly error workbook and checking that:

- annual NMBE remains close,
- monthly bias decreases,
- daily CVRMSE and MAPE improve,
- repeated constant fallback values disappear or become rare.
