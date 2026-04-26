"""
Data Loading and Preprocessing for PeriECG-RiskNet

Supports:
- PTB-XL dataset harmonization to 13 rhythm classes
- 7-lead reduction from 12-lead standard
- Signal quality checks and filtering
- Z-score normalization
- Resampling to target sampling rate

Usage:
    from src.preprocessing.loader import PTBXLDataModule, preprocess_ecg

    # For PTB-XL training
    dm = PTBXLDataModule(data_dir='data/ptbxl', batch_size=32)
    dm.setup()
    train_loader = dm.train_dataloader()

    # For single ECG inference
    ecg_processed = preprocess_ecg(raw_ecg, sampling_rate=500, target_rate=500)
"""

import os
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import wfdb
from scipy import signal
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard 12-lead order in PTB-XL
LEAD_NAMES_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# 7-lead subset for perioperative deployment
LEAD_NAMES_7 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1']
LEAD_INDICES_7 = [0, 1, 2, 3, 4, 5, 6]  # Indices in 12-lead order

# PTB-XL to 13-class rhythm mapping
RHYTHM_MAP_13 = {
    # Normal
    'NORM': 0,
    # Atrial fibrillation
    'AFIB': 1,
    # Atrial flutter
    'AFL': 2,
    # Supraventricular arrhythmias
    'SVARR': 3, 'PSVT': 3, 'SVTAC': 3,
    # Ventricular arrhythmias
    'VARR': 4, 'VTACH': 4, 'VFIB': 4,
    # Bradycardia
    'BRAD': 5, 'SBRAD': 5,
    # Tachycardia (non-sinus)
    'TACH': 6,
    # Bundle branch block
    'LBBB': 7, 'RBBB': 7, 'BBBB': 7, 'CLBBB': 7, 'CRBBB': 7,
    # Premature ventricular contraction
    'PVC': 8, 'VPB': 8,
    # Atrial premature contraction
    'APC': 9, 'APB': 9, 'NPAC': 9,
    # Paced rhythm
    'PACED': 10, 'PACE': 10,
    # Sinus tachycardia
    'STACH': 11,
    # Sinus bradycardia
    'SBRAD': 12,
}

# Additional mappings from diagnostic superclasses
DIAGNOSTIC_MAP = {
    'CD': [7],           # Conduction disturbance -> BBBB
    'STTC': [0],         # ST/T change -> Normal (or map to specific if needed)
    'MI': [0],           # Myocardial infarction -> Normal rhythm class
    'HYP': [0],          # Hypertrophy -> Normal rhythm class
    'NORM': [0],
}

TARGET_SR = 500          # Target sampling rate (Hz)
TARGET_LENGTH = 5000     # 10 seconds at 500 Hz
LOWCUT = 0.5             # High-pass filter cutoff (Hz)
HIGHCUT = 45.0           # Low-pass filter cutoff (Hz)
NOTCH_FREQ = 50.0        # Powerline frequency (Hz)
MAX_FLATLINE_RATIO = 0.3 # Maximum allowed flatline proportion


# ---------------------------------------------------------------------------
# Signal Processing Utilities
# ---------------------------------------------------------------------------

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Design Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def notch_filter(freq: float, fs: float, quality: float = 30.0):
    """Design notch filter for powerline interference."""
    b, a = signal.iirnotch(freq, quality, fs)
    return b, a


def remove_baseline_wander(ecg: np.ndarray, fs: float, cutoff: float = 0.5) -> np.ndarray:
    """Remove baseline wander using high-pass filter."""
    b, a = signal.butter(4, cutoff / (0.5 * fs), btype='high')
    return signal.filtfilt(b, a, ecg, axis=-1)


def remove_powerline_noise(ecg: np.ndarray, fs: float, freq: float = 50.0) -> np.ndarray:
    """Remove powerline interference using notch filter."""
    b, a = notch_filter(freq, fs)
    return signal.filtfilt(b, a, ecg, axis=-1)


def zscore_normalize(ecg: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-lead Z-score normalization."""
    mean = np.mean(ecg, axis=-1, keepdims=True)
    std = np.std(ecg, axis=-1, keepdims=True)
    return (ecg - mean) / (std + eps)


def resample_ecg(ecg: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample ECG to target sampling rate."""
    if orig_fs == target_fs:
        return ecg
    num_samples = int(ecg.shape[-1] * target_fs / orig_fs)
    return signal.resample(ecg, num_samples, axis=-1)


def check_signal_quality(ecg: np.ndarray, max_flatline_ratio: float = MAX_FLATLINE_RATIO) -> bool:
    """
    Check signal quality.
    Returns True if signal passes quality checks.
    """
    # Flatline detection: consecutive identical samples
    diff = np.diff(ecg, axis=-1)
    flatline_ratio = np.mean(np.abs(diff) < 1e-6, axis=-1).mean()
    if flatline_ratio > max_flatline_ratio:
        return False

    # Saturation detection
    if np.any(np.abs(ecg) > 10):  # Assuming normalized mV scale
        sat_ratio = np.mean(np.abs(ecg) > 10)
        if sat_ratio > 0.05:
            return False

    # Minimum amplitude check (avoid near-zero signals)
    if np.all(np.std(ecg, axis=-1) < 0.01):
        return False

    return True


def preprocess_ecg(ecg: Union[np.ndarray, torch.Tensor],
                   sampling_rate: int = 500,
                   target_rate: int = TARGET_SR,
                   target_length: int = TARGET_LENGTH,
                   leads: Optional[List[str]] = None,
                   apply_filter: bool = True,
                   normalize: bool = True) -> torch.Tensor:
    """
    Preprocess raw ECG for model input.

    Args:
        ecg: Raw ECG array. Shape: (leads, length) or (length, leads) or (batch, leads, length)
        sampling_rate: Original sampling rate in Hz
        target_rate: Target sampling rate
        target_length: Target signal length in samples
        leads: List of lead names if known (for reordering)
        apply_filter: Apply bandpass and notch filters
        normalize: Apply Z-score normalization

    Returns:
        Preprocessed ECG as torch.Tensor of shape (..., 7, target_length)
    """
    if isinstance(ecg, torch.Tensor):
        ecg = ecg.cpu().numpy()

    # Handle dimensions
    input_ndim = ecg.ndim
    if input_ndim == 1:
        ecg = ecg[np.newaxis, np.newaxis, :]  # (1, 1, L)
    elif input_ndim == 2:
        # Guess orientation: if first dim <= 12, assume (leads, length)
        if ecg.shape[0] <= 12:
            ecg = ecg[np.newaxis, :, :]  # (1, leads, L)
        else:
            ecg = ecg[np.newaxis, :, :]  # (1, leads, L) - may need transpose
            if ecg.shape[2] <= 12:  # Actually (L, leads)
                ecg = ecg.transpose(0, 2, 1)
    elif input_ndim == 3:
        pass  # (B, leads, L)
    else:
        raise ValueError(f"Unsupported ECG dimensions: {ecg.shape}")

    B, n_leads, L = ecg.shape

    # 7-lead reduction if 12 leads provided
    if n_leads == 12:
        ecg = ecg[:, LEAD_INDICES_7, :]
        n_leads = 7
    elif n_leads != 7:
        # Pad or truncate to 7 leads
        if n_leads < 7:
            pad = np.zeros((B, 7 - n_leads, L))
            ecg = np.concatenate([ecg, pad], axis=1)
        else:
            ecg = ecg[:, :7, :]
        n_leads = 7

    processed = []
    for i in range(B):
        sig = ecg[i].astype(np.float32)

        # Resample
        if sampling_rate != target_rate or L != target_length:
            sig = resample_ecg(sig, sampling_rate, target_rate)

        # Truncate or pad to target length
        if sig.shape[-1] > target_length:
            start = (sig.shape[-1] - target_length) // 2
            sig = sig[:, start:start + target_length]
        elif sig.shape[-1] < target_length:
            pad_width = target_length - sig.shape[-1]
            sig = np.pad(sig, ((0, 0), (0, pad_width)), mode='constant')

        if apply_filter:
            sig = remove_baseline_wander(sig, target_rate, LOWCUT)
            sig = remove_powerline_noise(sig, target_rate, NOTCH_FREQ)
            # Additional bandpass
            b, a = butter_bandpass(LOWCUT, HIGHCUT, target_rate)
            sig = signal.filtfilt(b, a, sig, axis=-1)

        if normalize:
            sig = zscore_normalize(sig)

        processed.append(sig)

    out = np.stack(processed, axis=0)  # (B, 7, target_length)
    return torch.from_numpy(out).float()


# ---------------------------------------------------------------------------
# PTB-XL Dataset
# ---------------------------------------------------------------------------

class PTBXLDataset(Dataset):
    """
    PTB-XL dataset with 13-class rhythm harmonization and 7-lead reduction.
    """
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 sampling_rate: int = 500,
                 target_length: int = 5000,
                 rhythm_classes: int = 13,
                 transform: Optional[callable] = None):
        """
        Args:
            data_dir: Path to PTB-XL root directory
            split: 'train', 'val', or 'test'
            sampling_rate: 500 or 100
            target_length: Target signal length
            rhythm_classes: Number of output classes (13)
            transform: Optional additional transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.rhythm_classes = rhythm_classes
        self.transform = transform

        # Load database
        self.df = self._load_database()
        self.label_map = self._build_label_mapping()

        # Filter by split
        strat_fold = self.df['strat_fold']
        if split == 'train':
            self.df = self.df[strat_fold <= 8]
        elif split == 'val':
            self.df = self.df[strat_fold == 9]
        elif split == 'test':
            self.df = self.df[strat_fold == 10]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.df = self.df.reset_index(drop=True)
        print(f"PTB-XL {split}: {len(self.df)} samples")

    def _load_database(self) -> pd.DataFrame:
        csv_path = self.data_dir / 'ptbxl_database.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"PTB-XL database not found at {csv_path}")
        df = pd.read_csv(csv_path, index_col='ecg_id')
        df['scp_codes'] = df['scp_codes'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        return df

    def _build_label_mapping(self) -> Dict[str, int]:
        """Build SCP code to 13-class ID mapping."""
        return RHYTHM_MAP_13

    def _scp_to_multilabel(self, scp_codes: dict) -> np.ndarray:
        """Convert SCP codes to multi-hot 13-class vector."""
        label = np.zeros(self.rhythm_classes, dtype=np.float32)
        for code, confidence in scp_codes.items():
            if code in self.label_map:
                cid = self.label_map[code]
                label[cid] = max(label[cid], float(confidence))
        # If no rhythm class matched, default to NORM if present
        if label.sum() == 0 and 'NORM' in scp_codes:
            label[0] = 1.0
        return label

    def _load_recording(self, filename: str) -> np.ndarray:
        """Load WFDB recording."""
        if self.sampling_rate == 500:
            path = self.data_dir / 'records500' / filename
        else:
            path = self.data_dir / 'records100' / filename

        record = wfdb.rdrecord(str(path))
        sig = record.p_signal.T.astype(np.float32)  # (leads, length)

        # Handle NaNs
        sig = np.nan_to_num(sig, nan=0.0)
        return sig

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        filename = row['filename_hr'] if self.sampling_rate == 500 else row['filename_lr']

        # Load signal
        sig = self._load_recording(filename)

        # Preprocess
        sig = preprocess_ecg(
            sig,
            sampling_rate=self.sampling_rate,
            target_rate=TARGET_SR,
            target_length=self.target_length
        )

        # Label
        label = self._scp_to_multilabel(row['scp_codes'])
        label = torch.from_numpy(label)

        if self.transform:
            sig = self.transform(sig)

        return sig.squeeze(0), label  # (7, 5000), (13,)


class ExternalDataset(Dataset):
    """
    Generic dataset for external validation cohorts.
    Expects .npz file with keys 'ecg' (N, 7, L) and optionally 'labels' (N,).
    """
    def __init__(self,
                 data_path: str,
                 target_length: int = 5000,
                 normalize: bool = True):
        data = np.load(data_path)
        self.ecgs = data['ecg']  # (N, 7, L)
        self.labels = data.get('labels', None)
        self.target_length = target_length
        self.normalize = normalize

        # Preprocess all
        self.processed = []
        for i in range(len(self.ecgs)):
            sig = preprocess_ecg(
                self.ecgs[i],
                sampling_rate=500,
                target_rate=TARGET_SR,
                target_length=target_length,
                normalize=normalize
            )
            self.processed.append(sig.squeeze(0))
        self.processed = torch.stack(self.processed)

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.processed[idx], torch.tensor(self.labels[idx], dtype=torch.long)
        return self.processed[idx]


# ---------------------------------------------------------------------------
# Data Module
# ---------------------------------------------------------------------------

class PTBXLDataModule:
    """Lightweight data module for PTB-XL."""
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 sampling_rate: int = 500):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_rate = sampling_rate

    def setup(self):
        self.train_ds = PTBXLDataset(self.data_dir, 'train', self.sampling_rate)
        self.val_ds = PTBXLDataset(self.data_dir, 'val', self.sampling_rate)
        self.test_ds = PTBXLDataset(self.data_dir, 'test', self.sampling_rate)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# CLI / Standalone
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess PTB-XL dataset')
    parser.add_argument('--data_dir', type=str, default='data/ptbxl')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--sampling_rate', type=int, default=500, choices=[100, 500])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading PTB-XL dataset...")
    dm = PTBXLDataModule(args.data_dir, batch_size=1, num_workers=0,
                         sampling_rate=args.sampling_rate)
    dm.setup()

    for split_name, dataset in [('train', dm.train_ds), ('val', dm.val_ds), ('test', dm.test_ds)]:
        print(f"
Processing {split_name} split ({len(dataset)} samples)...")
        ecgs = []
        labels = []
        for i in range(len(dataset)):
            ecg, label = dataset[i]
            ecgs.append(ecg.numpy())
            labels.append(label.numpy())

        ecgs = np.stack(ecgs)
        labels = np.stack(labels)

        out_path = os.path.join(args.output_dir, f'ptbxl_7lead_{split_name}_{args.sampling_rate}hz.npz')
        np.savez_compressed(out_path, ecg=ecgs, label=labels)
        print(f"Saved: {out_path} | Shape: {ecgs.shape}")

    print("
Preprocessing complete.")
