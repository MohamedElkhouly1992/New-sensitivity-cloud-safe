# Memory-Stable Execution Patch

This deployment patch changes only execution/output retention mechanics. It does not modify the dynamic RC equations, load equations, degradation model, control strategies, component-energy equations, comfort equations, or KPI definitions.

## Main changes
- Scenario timestep tables are appended directly to CSV in cloud memory-safe mode.
- Full scenario matrices are not retained and concatenated in RAM.
- Excel, PDF, figures, and secondary detailed tables are disabled by default during the run and generated only when requested.
- The Exports tab builds a summary Excel workbook plus full native and daily CSV.GZ files using chunked processing.
- Large download payloads are guarded to reduce Streamlit Community Cloud memory pressure.
- Temporary DataFrames are explicitly released after solver/export actions.

## Recommended Streamlit Cloud settings
- Python 3.12
- Main file: `streamlit_app.py`
- Keep **Cloud memory-safe execution** enabled.
- Run the solver first; build reports afterward from the Exports tab.
