"""
Inference Script for PeriECG-RiskNet

Supports:
- Single ECG file inference (.csv, .npy, .npz)
- Batch inference on directory of files
- Monte Carlo uncertainty estimation
- JSON output with risk probabilities and uncertainty scores
- Signal quality warnings

Usage:
    # Single ECG
    python scripts/inference.py \
        --input examples/sample_ecg.csv \
        --model_checkpoint periecgrisknet.pt \
        --output results.json \
        --mc_samples 50

    # Batch directory
    python scripts/inference.py \
        --input_dir data/external/ecgs/ \
        --model_checkpoint periecgrisknet.pt \
        --output_dir results/ \
        --batch_size 16
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.model import PeriECGRiskNet, build_model
from src.preprocessing.loader import preprocess_ecg, LEAD_NAMES_7
from src.utils.metrics import predictive_entropy, compute_alert_burden


# 13 rhythm class names
CLASS_NAMES = [
    "NORM", "AFIB", "AFL", "SVARR", "VARR", "BRAD", "TACH",
    "BBBB", "PVC", "APC", "PACED", "STACH", "SBRAD"
]


def load_ecg_from_file(filepath: str, sampling_rate: int = 500) -> Tuple[np.ndarray, int]:
    """
    Load ECG from various file formats.

    Returns:
        ecg: numpy array, shape (leads, length) or (length, leads)
        detected_sr: detected or provided sampling rate
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        df = pd.read_csv(filepath, header=None)
        ecg = df.values.astype(np.float32)
        # Auto-detect orientation
        if ecg.shape[0] <= 12 and ecg.shape[1] > ecg.shape[0]:
            ecg = ecg  # (leads, length)
        else:
            ecg = ecg.T
    elif suffix in ['.npy', '.npz']:
        if suffix == '.npy':
            ecg = np.load(filepath).astype(np.float32)
        else:
            data = np.load(filepath)
            if 'ecg' in data:
                ecg = data['ecg'].astype(np.float32)
            else:
                ecg = data[data.files[0]].astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return ecg, sampling_rate


def run_inference(model: PeriECGRiskNet,
                  ecg: torch.Tensor,
                  device: str = 'cuda',
                  mc_samples: int = 50,
                  return_attention: bool = False) -> Dict:
    """
    Run model inference with uncertainty quantification.

    Args:
        model: Loaded PeriECGRiskNet
        ecg: Preprocessed ECG tensor (1, 7, 5000)
        device: Computation device
        mc_samples: Number of MC dropout samples
        return_attention: Whether to return attention maps

    Returns:
        Dictionary with predictions and uncertainty
    """
    model.to(device)
    model.eval()

    ecg = ecg.to(device)

    # MC Dropout prediction
    mean_probs, uncertainty, std_probs = model.predict_with_uncertainty(
        ecg, mc_samples=mc_samples
    )

    mean_probs = mean_probs.cpu().numpy()[0]  # (13,)
    uncertainty = uncertainty.cpu().numpy()[0]  # scalar
    std_probs = std_probs.cpu().numpy()[0]  # (13,)

    # Top predictions
    top_indices = np.argsort(mean_probs)[::-1][:5]
    top_predictions = [
        {
            'class': CLASS_NAMES[idx],
            'class_id': int(idx),
            'probability': float(mean_probs[idx]),
            'std': float(std_probs[idx])
        }
        for idx in top_indices
    ]

    result = {
        'risk_probabilities': {
            CLASS_NAMES[i]: float(mean_probs[i]) for i in range(len(CLASS_NAMES))
        },
        'uncertainty': {
            'predictive_entropy': float(uncertainty),
            'confidence': float(1.0 - uncertainty),
            'interpretation': _interpret_uncertainty(uncertainty)
        },
        'top_predictions': top_predictions,
        'alert_recommendation': _alert_recommendation(mean_probs, uncertainty)
    }

    if return_attention:
        attn = model.get_attention_maps(ecg)
        result['attention_maps'] = {
            'lead_attention': attn['lead_attention'].cpu().numpy().tolist()
        }

    return result


def _interpret_uncertainty(uncertainty: float) -> str:
    """Provide clinical interpretation of uncertainty."""
    if uncertainty < 0.1:
        return "Low uncertainty: High confidence prediction suitable for automated alerting."
    elif uncertainty < 0.3:
        return "Moderate uncertainty: Consider clinical review before action."
    else:
        return "High uncertainty: Possible signal quality issue or out-of-distribution sample. Recommend repeat ECG."


def _alert_recommendation(probs: np.ndarray, uncertainty: float) -> Dict:
    """Generate alert recommendation based on risk and uncertainty."""
    max_risk = probs.max()
    risk_class = CLASS_NAMES[probs.argmax()]

    if uncertainty > 0.3:
        action = "MANUAL_REVIEW"
        reason = "High uncertainty requires clinician assessment."
    elif max_risk > 0.7:
        action = "HIGH_ALERT"
        reason = f"High probability of {risk_class} detected."
    elif max_risk > 0.5:
        action = "MODERATE_ALERT"
        reason = f"Elevated risk of {risk_class}."
    else:
        action = "NO_ALERT"
        reason = "No significant arrhythmia risk detected."

    return {
        'action': action,
        'reason': reason,
        'max_risk_class': risk_class,
        'max_risk_probability': float(max_risk)
    }


def process_single_file(model: PeriECGRiskNet,
                        filepath: str,
                        args) -> Dict:
    """Process a single ECG file."""
    try:
        raw_ecg, sr = load_ecg_from_file(filepath, args.sampling_rate)
    except Exception as e:
        return {
            'file': filepath,
            'error': str(e),
            'status': 'FAILED'
        }

    # Preprocess
    try:
        ecg_tensor = preprocess_ecg(
            raw_ecg,
            sampling_rate=sr,
            target_rate=500,
            target_length=5000,
            apply_filter=True,
            normalize=True
        )
    except Exception as e:
        return {
            'file': filepath,
            'error': f"Preprocessing failed: {str(e)}",
            'status': 'FAILED'
        }

    # Run inference
    result = run_inference(
        model, ecg_tensor,
        device=args.device,
        mc_samples=args.mc_samples,
        return_attention=args.return_attention
    )

    result['file'] = filepath
    result['status'] = 'SUCCESS'
    result['input_shape'] = list(raw_ecg.shape)
    result['sampling_rate'] = sr

    return result


def process_batch(model: PeriECGRiskNet,
                  filepaths: List[str],
                  args) -> List[Dict]:
    """Process multiple files."""
    results = []
    for fp in filepaths:
        result = process_single_file(model, fp, args)
        results.append(result)
        if args.verbose:
            status = result.get('status', 'UNKNOWN')
            print(f"[{status}] {fp}")
    return results


def main():
    parser = argparse.ArgumentParser(description='PeriECG-RiskNet Inference')

    # Input
    parser.add_argument('--input', type=str, default=None,
                        help='Path to single ECG file')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing ECG files')
    parser.add_argument('--pattern', type=str, default='*.csv',
                        help='File pattern for directory batch')

    # Model
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt or .pth)')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional JSON config for model architecture')

    # Inference settings
    parser.add_argument('--mc_samples', type=int, default=50,
                        help='Number of Monte Carlo dropout samples')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sampling_rate', type=int, default=500)
    parser.add_argument('--return_attention', action='store_true',
                        help='Return attention maps for interpretability')

    # Output
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output JSON file for single inference')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for batch inference')

    # Misc
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_probs', action='store_true',
                        help='Save full probability arrays')

    args = parser.parse_args()

    # Validate inputs
    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input_dir must be specified")

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Load model
    print(f"Loading model from {args.model_checkpoint}...")
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    model = build_model(config)
    checkpoint = torch.load(args.model_checkpoint, map_location=args.device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(args.device)
    model.eval()
    print("Model loaded successfully.")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Gather files
    if args.input:
        filepaths = [args.input]
        is_batch = False
    else:
        input_dir = Path(args.input_dir)
        filepaths = sorted(list(input_dir.glob(args.pattern)))
        if not filepaths:
            print(f"No files matching '{args.pattern}' found in {input_dir}")
            sys.exit(1)
        is_batch = True
        print(f"Found {len(filepaths)} files for batch inference")

    # Run inference
    print(f"Running inference (MC samples: {args.mc_samples})...")
    results = process_batch(model, filepaths, args)

    # Save results
    if is_batch:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'batch_results.json')
    else:
        output_path = args.output

    with open(output_path, 'w') as f:
        if is_batch:
            json.dump(results, f, indent=2)
        else:
            json.dump(results[0], f, indent=2)

    print(f"
Results saved to: {output_path}")

    # Summary
    successful = [r for r in results if r.get('status') == 'SUCCESS']
    failed = [r for r in results if r.get('status') == 'FAILED']
    print(f"Successful: {len(successful)} | Failed: {len(failed)}")

    if successful and args.verbose:
        print("
--- Sample Output ---")
        print(json.dumps(successful[0], indent=2))


if __name__ == '__main__':
    main()
