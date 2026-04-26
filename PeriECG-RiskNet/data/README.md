# Data Directory

This directory is intended for ECG datasets used in model development and validation.

## Directory Structure (after setup)

```
data/
├── README.md              # This file
├── ptbxl/                 # PTB-XL development cohort
│   ├── ptbxl_database.csv
│   ├── scp_statements.csv
│   ├── records500/        # 500 Hz recordings
│   │   ├── 00000/
│   │   ├── 00001/
│   │   └── ...
│   └── records100/        # 100 Hz recordings (optional)
├── external/              # External validation cohorts
│   ├── cohort_a/
│   └── cohort_b/
└── processed/             # Preprocessed outputs
    ├── ptbxl_7lead_500hz.npz
    ├── ptbxl_labels_13class.csv
    └── external_7lead.npz
```

## PTB-XL Setup

1. Download PTB-XL v1.0.3 from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
2. Extract to `data/ptbxl/`
3. Ensure `ptbxl_database.csv` and `records500/` are present

## 13-Class Harmonization

The original PTB-XL SCP-ECG statements are mapped to 13 rhythm classes:

| SCP Code | Rhythm Class | Class ID |
|----------|-------------|----------|
| NORM | Normal sinus rhythm | 0 |
| AFIB | Atrial fibrillation | 1 |
| AFL | Atrial flutter | 2 |
| SVARR / PSVT / SVTAC | Supraventricular arrhythmia | 3 |
| VARR / VTACH / VFIB | Ventricular arrhythmia | 4 |
| BRAD / SBRAD | Bradycardia | 5 |
| TACH / STACH | Tachycardia | 6 |
| LBBB / RBBB / BBBB | Bundle branch block | 7 |
| PVC / VPB | Premature ventricular contraction | 8 |
| APC / APB | Atrial premature contraction | 9 |
| PACED | Paced rhythm | 10 |
| (additional mappings) | Sinus tachycardia | 11 |
| (additional mappings) | Sinus bradycardia | 12 |

Run `src/preprocessing/loader.py` to perform harmonization automatically.

## 7-Lead Reduction

From standard 12-lead ECG, we extract:
- Lead I (index 0)
- Lead II (index 1)
- Lead III (index 2)
- aVR (index 3)
- aVL (index 4)
- aVF (index 5)
- V1 (index 6)

Leads V2–V6 are excluded for perioperative feasibility.

## External Cohorts

For external validation, prepare data as:
- NumPy array: `(N, 7, L)` where L = signal length
- Sampling rate: 500 Hz (or resample to 500 Hz)
- CSV label file with columns: `record_id, class_id, age, sex` (optional)

## Signal Quality Criteria

Preprocessing applies the following quality checks:
- Baseline wander removal (0.5 Hz high-pass filter)
- Powerline noise suppression (50/60 Hz notch)
- Amplitude normalization (Z-score per lead)
- Minimum 8s of analyzable signal
- Reject if >30% samples are flatline or saturation

## Data Privacy

External hospital cohorts are **not included** in this repository due to patient privacy regulations. Please contact the authors for data access inquiries.
