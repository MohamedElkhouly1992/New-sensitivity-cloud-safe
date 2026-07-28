# DataFrame Defragmentation Patch Report

## Purpose
This patch improves the stability and memory efficiency of the daily export step in the dynamic RC core solver bundle.

## What was changed
In `hvac_v3_engine.py`, the daily aggregation function was patched near the daily export section.

The original code created the grouped daily DataFrame and then inserted several new columns one by one:

- `energy_kwh_day`
- `co2_kg_day`
- `cost_usd_day`
- `daily_component_balance_kwh`
- `daily_balance_error_kwh`

Pandas warned that the DataFrame was highly fragmented. The patched version now:

1. Calls `daily = daily.copy()` immediately after daily groupby aggregation.
2. Builds export-only columns in a temporary dictionary.
3. Adds them together using `pd.concat(...)`.
4. Calls `daily = daily.copy()` again after concatenation.

## Scientific effect
This patch does **not** change the dynamic RC solver equations, HVAC energy calculation, degradation calculation, strategy logic, or scenario outputs.

It only changes the way the daily export DataFrame is assembled in memory.

## Expected effect
- Lower risk of Streamlit Cloud memory/performance warnings.
- Faster daily export generation.
- More stable Excel/CSV export for large scenario runs.
- No change in calculated energy, comfort, CO2, degradation, or BHI values.
