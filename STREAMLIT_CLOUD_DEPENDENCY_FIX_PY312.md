# Streamlit Cloud dependency fix — Python 3.12

## Cause of the failed deployment

The previous dependency file pinned `numpy==2.3.2` while also installing `catboost==1.2.7`. CatBoost 1.2.7 requires `numpy>=1.16,<2.0`, so the dependency resolver correctly rejected the environment as unsatisfiable.

## Applied correction

The deployment profile now uses:

```text
streamlit>=1.59,<1.61
pandas==2.3.3
numpy==1.26.4
matplotlib==3.11.1
scikit-learn==1.9.0
openpyxl==3.1.5
plotly>=6.0,<7
catboost==1.2.7
shap==0.46.0
```

`numpy==1.26.4` satisfies CatBoost's `<2.0` constraint and also satisfies the minimum NumPy requirements of pandas, Matplotlib, and scikit-learn in this package.

## Deployment settings

Deploy the repository with Python 3.12. The package includes both `runtime.txt` and `.python-version` set to Python 3.12 for consistency, although the Python version selected in Streamlit Community Cloud deployment settings remains authoritative.

## Required repository-root files

Keep these files at the repository root:

- `streamlit_app.py`
- `hvac_v3_engine.py`
- `integrated_sensitivity_suite.py`
- `report_addons.py`
- `requirements.txt`
- `runtime.txt`

After replacing the files, commit the changes and reboot the Streamlit app. If Streamlit Cloud continues to use a cached failed environment, delete and redeploy the app with Python 3.12 selected in Advanced settings.
