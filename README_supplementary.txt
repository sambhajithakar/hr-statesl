Supplementary Materials
Machine learning cannot recover what pooling removes: genetic context bounds
synthetic lethal target discovery in breast cancer
S. B. Thakar and P. N. Ghorpade

Contents
  HR-StateSL-edges-v1.0.csv   the curated, evidence-typed, state-conditioned edge resource
  code/01_resource_stats.py   computes every count reported in Section 3.6 from the resource
  code/02_theory_and_simulation.py   closed-form attenuation, the learning simulation and the design calculation
  code/03_estimation_extended.py     the estimation simulation and its parameter sweeps
  code/04_metric_blindness.py        precision and class-resolved detection over 500 replicates
  code/05_figures.py                 draws Figures 1 and 2 from the stored results
  code/06_verify_manuscript_numbers.py  re-checks every numeric claim against the stored results
  results/*.json              the stored results that the manuscript and figures read

Reproducing the results
  Run the scripts in numerical order from the directory containing code/ and results/.
  Random seeds are the integers from 0 to the replicate count, set through
  numpy.random.default_rng, so every reported value reproduces exactly.
  Software versions used: Python 3, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1,
  pandas 3.0.5, Matplotlib 3.11.2.
  06_verify_manuscript_numbers.py prints ALL CHECKS PASSED when every manuscript
  number matches the stored results.

No new experimental data were generated. No patient data were used.
