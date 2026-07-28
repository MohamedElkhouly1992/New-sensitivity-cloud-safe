# Performance Warning Cleanup Patch

This patch is applied to the dynamic RC core solver deployment bundle.

## Purpose

The previous defragmentation patch reduced repeated column insertion, but Streamlit Cloud still showed pandas `PerformanceWarning: DataFrame is highly fragmented` during the daily aggregation line. This warning was generated during export-table construction, not during the physical HVAC solver calculation.

## Changes made

1. Added explicit DataFrame consolidation before daily aggregation.
2. Wrapped only the daily export groupby aggregation in a local pandas `PerformanceWarning` suppression block.
3. Added explicit consolidation after daily aggregation and after export-only alias/balance columns are appended.
4. Cleaned the `use_hvac_preset` Streamlit session-state warning by avoiding simultaneous `value=` and `key=` assignment for the same widget.

## What is not changed

The patch does not modify dynamic RC equations, degradation equations, energy calculation, comfort calculation, strategy logic, monthly correction factors, or scenario outputs. It only changes memory handling and warning cleanup in the export/UI layer.
