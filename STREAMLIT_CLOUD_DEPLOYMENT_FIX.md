# Streamlit Community Cloud deployment fix

This package now targets **Python 3.12**.

The most recent failed deployment was caused by an unsatisfiable dependency pair:

- `numpy==2.3.2`
- `catboost==1.2.7`, which requires `numpy<2.0`

The corrected `requirements.txt` uses `numpy==1.26.4`, preserving CatBoost and SHAP support while satisfying the NumPy minimum versions required by the remaining scientific packages.

See `STREAMLIT_CLOUD_DEPENDENCY_FIX_PY312.md` for the full deployment steps.
