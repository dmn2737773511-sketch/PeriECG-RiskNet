# PeriECG-RiskNet

Uncertainty-Aware Deep Learning Framework for Perioperative Arrhythmia Risk Stratification using Reduced-Lead ECG

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the official implementation of the paper:

> **PeriECG-RiskNet: An Uncertainty-Aware Reduced-Lead ECG Framework for Perioperative Arrhythmia Risk Stratification**

## Overview

PeriECG-RiskNet is a translational deep learning framework designed for preoperative arrhythmia risk stratification. The system integrates:

- **Reduced-lead ECG modeling**: A clinically feasible 7-lead configuration (I, II, III, aVR, aVL, aVF, V1) derived from standard 12-lead ECG
- **Lead-aware CNN-iTransformer-LSTM architecture**: Captures spatial lead correlations, temporal dynamics, and long-range dependencies
- **Uncertainty quantification**: Monte Carlo Dropout-based epistemic uncertainty for calibrated risk assessment
- **Wearable-compatible pipeline**: Supports signal acquisition from hydrogel-assisted wearable electrodes (ADS1298-based)

## Repository Structure

```
PeriECG-RiskNet/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore rules
│
├── data/
│   └── README.md            # Data preparation instructions
│
├── src/
│   ├── models/
│   │   └── model.py         # Lead-aware CNN-iTransformer-LSTM model
│   ├── preprocessing/
│   │   └── loader.py        # Data loading, PTB-XL harmonization, 7-lead reduction
│   └── utils/
│       └── metrics.py       # Evaluation metrics and uncertainty calibration
│
├── scripts/
│   └── inference.py         # Inference script for single ECG or batch processing
│
└── examples/
    └── sample_ecg.csv       # Example 7-lead ECG snippet for quick testing
```

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA-capable GPU (recommended for training)

### Setup

```bash
git clone https://github.com/yourusername/PeriECG-RiskNet.git
cd PeriECG-RiskNet
pip install -r requirements.txt
```

## Data Preparation

### PTB-XL Development Cohort

1. Download PTB-XL from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
2. Place `ptbxl_database.csv` and `records500/` (or `records100/`) in `data/ptbxl/`
3. Run harmonization to 13 rhythm classes (see `data/README.md`)

### 7-Lead Configuration

The model uses a reduced-lead subset compatible with perioperative workflows:
- **Limb leads**: I, II, III
- **Augmented leads**: aVR, aVL, aVF
- **Precordial lead**: V1

Lead reduction is performed automatically in the data loader via `LEAD_INDICES = [0, 1, 2, 3, 4, 5, 6]` mapping from standard 12-lead order.

### External Validation Cohorts

For external validation, prepare ECGs as NumPy arrays of shape `(N, 7, L)` where:
- `N` = number of recordings
- `7` = number of leads (I, II, III, aVR, aVL, aVF, V1)
- `L` = signal length (default: 5000 samples at 500 Hz for 10s)

## Quick Start

### 1. Single ECG Inference

```bash
python scripts/inference.py     --input examples/sample_ecg.csv     --model_checkpoint periecg_risknet.pt     --output results.json     --mc_samples 50
```

### 2. Batch Inference

```python
from src.models.model import PeriECGRiskNet
from src.preprocessing.loader import preprocess_ecg
import torch

# Load model
model = PeriECGRiskNet(num_classes=13, num_leads=7, mc_dropout=True)
model.load_state_dict(torch.load('periecgrisknet.pt'))
model.eval()

# Preprocess ECG (N, 7, 5000)
ecg_tensor = preprocess_ecg(raw_ecg_array, sampling_rate=500)

# Uncertainty-aware inference
with torch.no_grad():
    logits, uncertainty = model(ecg_tensor, return_uncertainty=True)
    probs = torch.softmax(logits, dim=-1)

# probs: (N, 13) risk probabilities
# uncertainty: (N,) epistemic uncertainty scores
```

### 3. Training (PTB-XL)

```bash
python -m src.preprocessing.loader --data_dir data/ptbxl --output_dir data/processed
# Then use your training loop with src/models/model.py
```

## Model Architecture

```
Input: (Batch, 7 leads, 5000 samples)
  ↓
1D CNN Encoder (per-lead feature extraction)
  ↓
Lead-Aware Spatial Attention (cross-lead correlation)
  ↓
iTransformer Encoder (inter-lead temporal dependencies)
  ↓
LSTM Temporal Aggregator (sequential dynamics)
  ↓
Fully-Connected Classifier + MC Dropout
  ↓
Output: 13-class risk probabilities + uncertainty estimate
```

### Key Components

| Component | Description |
|-----------|-------------|
| `CNNBackbone` | Multi-scale 1D convolutions with residual connections |
| `LeadAwareAttention` | Cross-lead attention mechanism for spatial ECG patterns |
| `iTransformerBlock` | Inverted Transformer treating leads as tokens, time as features |
| `LSTMAggregator` | Bidirectional LSTM for temporal risk evolution |
| `UncertaintyHead` | Monte Carlo Dropout for epistemic uncertainty |

## Rhythm Classes (13-class harmonization)

| ID | Class | Description |
|----|-------|-------------|
| 0 | NORM | Normal sinus rhythm |
| 1 | AFIB | Atrial fibrillation |
| 2 | AFL | Atrial flutter |
| 3 | SVARR | Supraventricular arrhythmia |
| 4 | VARR | Ventricular arrhythmia |
| 5 | BRAD | Bradycardia |
| 6 | TACH | Tachycardia |
| 7 | BBBB | Bundle branch block |
| 8 | PVC | Premature ventricular contraction |
| 9 | APC | Atrial premature contraction |
| 10 | PACED | Paced rhythm |
| 11 | STACH | Sinus tachycardia |
| 12 | SBRAD | Sinus bradycardia |

## Uncertainty Interpretation

The model outputs **epistemic uncertainty** via Monte Carlo Dropout (50 forward passes):

- **Low uncertainty** (< 0.1): High confidence prediction, suitable for automated alerting
- **Moderate uncertainty** (0.1–0.3): Consider clinical review before action
- **High uncertainty** (> 0.3): Signal quality issue or out-of-distribution sample; recommend repeat ECG

Uncertainty is computed as the **predictive entropy** of MC samples:

```
H[y|x,D] = -Σ_c (1/T Σ_t p_c^t) log(1/T Σ_t p_c^t)
```

## Performance Benchmarks

### PTB-XL 7-Lead Subset

| Metric | Value |
|--------|-------|
| AUC-ROC (macro) | 0.94 |
| AUC-PR (macro) | 0.89 |
| F1-score (macro) | 0.82 |
| Calibration (ECE) | 0.031 |

*Note: Full benchmark results and external validation statistics are reported in the manuscript.*

## Hardware & Signal Acquisition

The framework is validated with a custom wearable platform:
- **Electrodes**: Hydrogel-assisted Ag/AgCl electrodes (low-bias conductance)
- **ADC**: Texas Instruments ADS1298 (24-bit, 8-channel, 500 Hz)
- **Transmission**: BLE 5.0 to workstation
- **Posture robustness**: Validated across supine, lateral, and sitting positions

Signal quality checks are integrated in `src/preprocessing/loader.py`.

## Citation

If you use this code or framework in your research, please cite:

```bibtex
@article{periecgrisknet2025,
  title={Uncertainty-Aware Reduced-Lead ECG Framework for Perioperative Arrhythmia Risk Stratification},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

For questions regarding code or reproduction, please open an issue on GitHub or contact the corresponding author.

## Acknowledgments

- PTB-XL dataset: Wagner et al., *Scientific Data*, 2020
- iTransformer architecture: Liu et al., *ICLR*, 2024
