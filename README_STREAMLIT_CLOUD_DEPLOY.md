# HVAC v3 Dynamic RC Core Solver — Memory-Stable Streamlit Cloud Deployment

## Deploy
1. Upload all files in this folder to the root of a GitHub repository.
2. In Streamlit Community Cloud choose:
   - Main file: `streamlit_app.py`
   - Python: `3.12`
3. Deploy or reboot the app.

## Recommended run sequence
1. Open **Scenario modeling**.
2. Keep **Cloud memory-safe execution** enabled.
3. Keep Excel, PDF, figures, and detailed secondary outputs disabled during the solver run.
4. Run the model.
5. Open **Exports and results** and click **Build cloud-safe export package**.

The package contains:
- `results_summary.xlsx` — metadata, summary, annual results, and bounded previews.
- `native_timestep_data.csv.gz` — full native solver timestep/hourly output.
- `daily_data.csv.gz` — full daily aggregation.
- `cloud_safe_results_package.zip` — combined package.

## Numerical integrity
Memory-safe execution changes only how scenario result tables are retained and written. The dynamic RC solver equations, degradation model, control strategies, energy calculations, comfort calculations, and KPI definitions are unchanged.
