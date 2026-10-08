# Descriptive reporting addendum

After the first frozen computation completed on 2026-10-06, a presentation review requested matched-quartet equal weighting alongside the already computed success/failure-pair weighted concordance. This addendum changes no inputs, target selection, detectors, features, model fitting, predictions or failure labels.

For each mixed quartet, pair concordance is the mean of the success-versus-failure pair ordering credits, with 0.5 for a risk tie. Quartet-equal concordance averages this measure once per mixed quartet. Pair-weighted concordance pools the ordering credits across all success/failure pairs. Both are retained in `quartet_summary.csv` and `analysis_summary.json`.

The same presentation review requested clean baseline median/range and energy/amplitude failure counts, patient-equal median/range AUROC summaries, and explicit source-holdout checks. These are descriptive summaries of the completed run, with no inferential confidence interval or p-value. They are in `descriptive_verification.json`, `patient_equal_metric_summary.csv` and `source_holdout_checks.json`.

The fixed 0.10 risk policy is additionally tabulated by patient for all three fitted feature sets in `fixed_threshold_by_patient.csv`. `fixed_threshold_source_counts.json` separates failures originating from clean-baseline-success and clean-baseline-failure targets and counts excluded nonfailures. These are decompositions of the originally prespecified policy; the risk threshold is unchanged.
