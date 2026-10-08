# Event-local assessment of beat-detection reliability in wearable electrocardiography

This directory contains the reproducible computation for the Journal of Medical Systems manuscript. It is a separate research analysis within PeriECG-RiskNet, with its own inputs, endpoint and dependencies. Its endpoint is detector failure in a 20-second ECG window, not diagnosis or prediction of a clinical event.

## Install

Use Python 3.12, the version tested for this release. From this directory run:

```bash
python -m pip install -r requirements.txt
```

The numerical rerun used NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0. The listed versions reproduce the saved predictions and fitted parameters. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before running on Unix-like systems. On Windows these can be set through the usual environment-variable controls.

## Primary wearable archive: numeric reproduction

```bash
python run_analysis.py metrics --out reanalysis
```

The primary computation has 16,384 replay episodes from one coded recording series under four sequential acquisition conditions. Joint target/residual source exclusion produces four folds with 1,024 test episodes per fold; the pooled diagonal source-excluded evaluation has 4,096 distinct predictions. These conditions are recording sources, not four people. The supplied feature matrix has 16,384 rows and 30 columns. The command refits the 12 reliability models and recomputes prediction, calibration, retention, conditional quartet ranking and fixed-threshold tables.

`source_data/conditional_alignment` supplies complete mixed-quartet rankings and retained/excluded failure counts. `source_data/sensitivity_analyses.csv` and the saved verification reports document the original full waveform computation. Numeric mode does not rebuild residuals, rerun detectors, or recompute waveform-based sensitivity checks.

For an authorized user who has the wearable replay bank:

```bash
python run_analysis.py replay --bank /path/to/replay_banks.npz --out waveform_rerun
```

For an authorized user with the four original source CSVs or their archives:

```bash
python run_analysis.py full --raw-dir /path/to/authorized_inputs --out full_rerun
```

The private wearable waveforms, processed caches, exact acquisition filenames, operator metadata and replay bank are excluded from this repository. The public source manifest retains source codes, sample totals and hashes. Access to original recordings remains subject to author and institutional governance.

## Public annotated ECG: numeric reproduction

```bash
python public_validation/refit_public_models.py --out refit_results
```

This refits all 141 public reliability models from the included 18,432-by-19 feature matrix and complete outcome table. It reproduces patient-held-out predictions and evaluation tables without downloading raw ECG. The saved descriptive addendum can be recomputed with `python public_validation/summarize_completed_run.py`; that command reads the supplied results in `public_validation`, not the separate `refit_results` directory.

## Public annotated ECG: full reconstruction

```bash
python public_validation/download_public_data.py
python public_validation/run_public_validation.py
python public_validation/summarize_completed_run.py
```

The downloader checks each file against the included frozen SHA-256 manifest. The full run uses all 48 MIT-BIH records, 47 patient groups and 192 continuous annotated targets, with six fixed NSTDB noise windows. Records 201/202 form one patient group. All clean-baseline failures remain included. The 47 patient folds share the known noise bank. The two-channel models are refitted separately; this experiment does not externally test the previously fitted three-channel wearable model. `FROZEN_PLAN.md` and `REPORTING_ADDENDUM.md` document the design and reporting sequence.

## Figures

```bash
python additional_figures/plot_Fig_S31.py
python additional_figures/plot_Fig_S32.py
```

These commands regenerate the two new supplementary figures from the supplied tables. The local, original `qa_tools/panel_alignment.py` checks the physical two-by-two panel geometry and labels. Scripts use Arial when available and otherwise fall back to installed sans-serif fonts; the exact image bytes can therefore differ across systems, while plotted values are unchanged.

The seven original main plates and original S1/S2 spectral plates require authorized waveform inputs:

```bash
python reproduce_figures.py --raw-cache /path/to/processed_caches --bank /path/to/replay_banks.npz --out figures
```

This does not regenerate the other archived hardware figures or the full supporting document. Those documents and photos are not repository assets.

## Checks and reuse

```bash
python verify_release.py
```

This verifies all included file hashes and confirms the two feature arrays contain only numeric features and names. The supplied numeric preflight records byte-identical primary/public predictions and fitted parameters, with identical evaluation values. Confidence intervals or p-values based on independent replay episodes are not claimed. Fixed risk thresholds report retained failures and exclusion costs; they do not guarantee patient-specific risk control.

The repository's existing MIT license covers its original software. The independently authored panel geometry checker carries the same license. Dataset rights are separate. MIT-BIH and NSTDB inputs are attributed to PhysioNet under Open Data Commons Attribution License v1.0; versions, DOIs, original publications and URLs are recorded in `public_validation/SOURCES.json`. Cite the relevant original dataset publications and the manuscript when reusing this analysis. The numerical tables are supplied as research replication material, with source governance retained for the wearable archive.
