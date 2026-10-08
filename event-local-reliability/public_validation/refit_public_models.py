"""Refit public models from supplied numeric features and outcomes, without raw download."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
import run_public_validation as analysis
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=Path('refit_results'))
args=parser.parse_args()
source=Path(__file__).resolve().parent
rows=list(csv.DictReader((source/'replay_outcomes.csv').open()))
integer={'start_seconds','noise_start_seconds','shift_samples','baseline_primary_success','tp','fp','fn','failure','amplitude_tp','amplitude_fp','amplitude_fn'}
floating={'snr_db','clean_f1','clean_f1_amplitude','noise_rms_mV','f1','f1_amplitude'}
for row in rows:
    for key in integer:row[key]=int(row[key])
    for key in floating:row[key]=float(row[key])
z=np.load(source/'reliability_features.npz',allow_pickle=False)
analysis.ROOT=args.out.resolve();analysis.ROOT.mkdir(parents=True,exist_ok=True)
result=analysis.evaluate(rows,z['X'],list(z['names']))
(analysis.ROOT/'refit_summary.json').write_text(json.dumps(result,indent=2))
print('Completed:',analysis.ROOT)
