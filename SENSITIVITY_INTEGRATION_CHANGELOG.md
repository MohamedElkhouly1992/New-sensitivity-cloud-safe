# Sensitivity Integration Changelog

- Added the `Comprehensive Sensitivity Studio` as tab 24 without removing existing tabs.
- Added official-solver Local OAT, paired Monte Carlo, Morris, Sobol, benchmark, ablation, optimizer robustness, and connectivity analyses.
- Added separate CSV/PNG/SVG/JSON/ZIP extraction for each analysis.
- Added the sensitivity output directory to the existing Run Result Collector.
- Added an editable parameter-range table and CSV template.
- Confirmed that the new module has no dependency on `ChillerPlant.csv`.
- Preserved the existing dynamic RC solver, EMS, validation, report, and deployment files from the supplied Version 5 package.


## Session-state hotfix
- Removed reassignment of the instantiated `integrated_sensitivity_out_dir` widget key.
- Added `integrated_sensitivity_results_dir` for the derived comprehensive-sensitivity folder.
- Added the derived folder to the Run Result Collector search paths.
