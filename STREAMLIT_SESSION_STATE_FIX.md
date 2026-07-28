# Streamlit Session-State Fix

## Error corrected

`st.session_state.integrated_sensitivity_out_dir cannot be modified after the widget with key integrated_sensitivity_out_dir is instantiated.`

## Cause

The shared weather-control helper creates the output-folder text box with widget key `integrated_sensitivity_out_dir`. The sensitivity tab then appended `comprehensive_sensitivity` and attempted to write the derived path back to the same key during the same Streamlit run. Streamlit protects instantiated widget keys from direct reassignment.

## Correction

- The widget value remains stored in `integrated_sensitivity_out_dir`.
- The derived results path is stored separately in `integrated_sensitivity_results_dir`.
- The Run Result Collector now scans both paths.
- Solver equations, parameter ranges, sampling algorithms, and result calculations are unchanged.

## Deployment

Replace the repository files with this package, commit, push, and reboot the Streamlit app.
