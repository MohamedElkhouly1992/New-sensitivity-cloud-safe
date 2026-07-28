# Comprehensive Local and Global Sensitivity Studio

## Scope

The new **Comprehensive Sensitivity Studio** is integrated as a separate tab in the existing HVAC ROM-Degradation Streamlit application. Every evaluation calls the same `simulate_combo()` numerical pathway used by Scenario Modeling. It does not use `ChillerPlant.csv`, a surrogate model, or a separate simplified HVAC equation set.

## Included analyses

1. **Local OAT sensitivity**
   - Baseline, lower-bound, and upper-bound simulation for every enabled parameter.
   - Signed low/high percentage effects.
   - Central elasticity.
   - A separate CSV, PNG, and SVG tornado figure for every KPI.

2. **Paired Monte Carlo robustness**
   - Uses the same uncertain parameter realization for S0, S1, S2, and S3.
   - Produces uncertainty summaries, strategy-ranking probabilities, and paired S3-versus-S0 results.
   - Can fix the optimizer seed to isolate physical uncertainty.

3. **Morris global screening**
   - Reports `mu`, `mu_star`, and `sigma` together with normalized forms.
   - Intended to reduce the full parameter set before Sobol analysis.

4. **Sobol global sensitivity**
   - First-order `S1` and total-order `ST` indices.
   - Bootstrap 95% confidence intervals.
   - Reports `ST-S1` as an interaction indicator.
   - Finite-sample negative values or values above one are retained rather than clipped.

5. **Benchmark cases**
   - No degradation.
   - Nominal condition.
   - Severe degradation.
   - High infiltration.

6. **S3 ablation**
   - S0 reactive baseline.
   - Full S3.
   - S3 control only.
   - S3 maintenance only.
   - S3 without degradation feedback in the objective.

7. **Optimizer robustness**
   - Population size, iteration count, and random-seed matrix.
   - Reports mean, standard deviation, range, and coefficient of variation of energy.

8. **Parameter connectivity**
   - Confirms that each enabled uncertain parameter reaches at least one official model KPI.

## Separate extraction

Every analysis produces separate download buttons for:

- raw samples;
- sensitivity indices;
- summary tables;
- strategy rankings;
- PNG figures;
- editable SVG figures;
- metadata JSON;
- one ZIP containing all outputs for that analysis.

The **Export Index** sub-tab lists every generated result separately. The main **Run Result Collector** also detects the sensitivity output folder.

## Recommended workflow

1. Use one simulation year and daily time step for a diagnostic connectivity test.
2. Run Local OAT across all enabled parameters.
3. Run Morris with 10 trajectories for a diagnostic test, then 20–30 trajectories for manuscript results.
4. Select the 8–12 most influential parameters.
5. Run Sobol with `N=32–64` diagnostically, then `N=256–1024` when computationally feasible.
6. Run paired Monte Carlo with at least 100 samples for the final strategy-ranking assessment.
7. Run S3 ablation and optimizer robustness separately.

## Computational caution

The official S3 solver performs stochastic candidate evaluation at every time step. Global analysis can therefore be expensive. The tab displays the expected number of complete official solver evaluations before execution. Start with small diagnostic settings before using publication-level sample sizes.

## Main files

- `streamlit_app.py`: existing application with the new tab.
- `hvac_v3_engine.py`: official dynamic RC/degradation engine.
- `integrated_sensitivity_suite.py`: local/global sensitivity algorithms and tab renderer.
- `sensitivity_parameter_ranges_template.csv`: editable default uncertainty ranges.
