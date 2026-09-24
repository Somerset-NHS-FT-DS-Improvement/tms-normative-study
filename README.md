# Analysis Code for Reference Limits Using Transcranial Magnetic Stimulation in Healthy Adults

## Content

- `01_data_cleaning.ipynb` - Cleaning raw TMS data, checking for missing values and exporting analysis ready datasets.
- `02_cohort_and_descriptives.ipynb` - Storing cohort summaries and high level descriptive analysis.
- `03_reference_limits.ipynb` - Estimate 2.5th, 50th, and 97.5th percentile reference limits with bootstrap-derived confidence intervals and measures of asymmetry.
- `04_regression.ipynb` - Estimate height-adjusted reference limits (97.5th percentile) for CMCT using linear regression and empirical residual distributions.
