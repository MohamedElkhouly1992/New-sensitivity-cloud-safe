# Streamlit Cloud Stable Deployment Notes

This patch is for deployment stability only. It does not modify the dynamic RC solver, degradation equations, energy balance, comfort calculation, or scenario logic.

Changes included:

1. Pinned requirements to avoid Streamlit Cloud selecting very new Python/package combinations automatically.
2. Added `.streamlit/config.toml` with port/address/upload settings.
3. Suppressed non-fatal Streamlit UI warning logs.
4. Kept the DataFrame defragmentation/export patch.

Important deployment action:

When deploying on Streamlit Community Cloud, choose Python 3.11 or 3.12 from Advanced settings. The log must not show Python 3.14 for this research bundle.

Main file path:

```text
streamlit_app.py
```

If the app still fails the `/healthz` check after a heavy run, use local PC deployment for full 5-year/hourly runs and keep Streamlit Cloud only for small demos.
