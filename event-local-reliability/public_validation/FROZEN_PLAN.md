# Frozen public ECG computation plan

Frozen on 2026-10-06 before downloading or inspecting ECG performance. The existing manuscript detector and scoring implementation will be imported unchanged from `recovered/analysis_code/run_stress.py`. This study uses archived public benchmark waveforms and annotations, not prospective clinical validation and not validation of the already fitted three-channel S01 models.

## Data and target windows

- All 48 MIT-BIH Arrhythmia Database version 1.0.0 records; 47 patient groups. Records 201 and 202 are bound to one group.
- Preserve the original first and second channel order, including reversed leads and paced records. The first channel is the primary detector channel. No record or window is excluded because of detector performance.
- Four contiguous 20 s target windows per record, beginning at 60, 120, 180 and 240 s. Resample physical mV signals from 360 to 500 Hz with `scipy.signal.resample_poly(up=25, down=18)` across the full recording. Apply the existing fourth-order 0.5–40 Hz Butterworth zero-phase filter to the full resampled recording before selecting windows.
- Beat reference symbols are N, L, R, B, A, a, J, S, V, r, F, e, j, n, E, /, f, Q and ?. Rhythm markers, non-conducted P waves, flutter waves and isolated QRS-like artifacts are excluded from beat references. Annotation locations are rounded after the 25/18 conversion and used only for outcome scoring.
- Score samples 1000 through 8999 (2–18 s), using the imported one-to-one greedy matching with 38 sample tolerance. Primary failure is F1 < 0.95. Both fixed energy and amplitude detector clean scores are retained.

## Fixed perturbation bank

- MIT-BIH Noise Stress Test Database version 1.0.0 `bw`, `ma`, `em` two-channel noise records, preserving channel order.
- Resample and filter as above. Two 20 s windows per noise type beginning at 0 and 60 s. Demean, apply the 10000-point Hann taper, demean again. Normalize the two-channel RMS with the manuscript's 1e-12 safeguard.
- Target-to-residual RMS ratios 20, 10, 0 and −5 dB; joint circular rotations 0, 69, 196 and 367 samples, giving 192 × 6 × 4 × 4 = 18432 episodes in 4608 matched quartets.
- Check residual energy, complete FFT power and cross-channel covariance invariance numerically. The corrupted signal spectrum is not assumed invariant.

## Features and held-out evaluation

- Global features: the existing six features per channel (three Welch fractions, log SD, kurtosis and normalized 99th percentile difference) plus one cross-channel Pearson correlation, total 13.
- Event features: first-to-second channel energy detector agreement, first-channel energy-to-amplitude detector agreement, and median/10th percentile beat-template cosine per channel, total 6. All morphology snippets are located by first-channel energy detections. Combined total 19.
- Untrained agreement ranking is `1 - min(two agreements)`.
- Train separate Global, Event and Combined models in leave-one-patient-group-out evaluation, with `StandardScaler` fitted only on training episodes and logistic regression C=1, L2, lbfgs, max_iter=2000, random_state=20261003. No hyperparameter search, resampling, class weighting or recalibration.
- Target patient groups are held out. The fixed six noise windows are shared between training and test. This tests unseen patient target waveforms under a known perturbation bank, not simultaneous independence of every source component.
- Report pooled out-of-fold AUROC, AP, Brier, ten equal-width bin ECE, fixed predicted-risk <= 0.10 retention, and ranking coverage 50%, 70%, 80%, 100%. Retained-count rounding and sorting follow the existing workflow. No episode-independent confidence intervals or p-values.
- Report matched quartet label crossings, invariant minimum error and constant-probability Brier floor on all targets and separately on targets whose clean primary F1 >= 0.95. Report mixed-quartet success/failure pair concordance as a mechanism-specific ranking measure.

## Outcome-independent reporting

Retain every window, clean baseline and corrupted outcome. Do not change the plan in response to performance. Any software correction will be recorded with a reason. The first completed run determines the reported results.

## Official sources checked

- https://physionet.org/content/mitdb/1.0.0/ confirms 48 two-channel 360 Hz records from 47 subjects with cardiologist beat annotations and Open Data Commons Attribution License v1.0.
- https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm confirms records 201 and 202 share a patient, variable lead configurations and inclusion of paced records.
- https://physionet.org/content/nstdb/1.0.0/ confirms the bw/ma/em recordings, recording context and Open Data Commons Attribution License v1.0.
