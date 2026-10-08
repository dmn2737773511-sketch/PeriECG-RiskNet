#!/usr/bin/env python3
"""Reproduce the supplied computational analysis; no external services required."""
from pathlib import Path
import argparse, os, shutil, subprocess, sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['metrics','replay','full'])
    parser.add_argument('--out',type=Path,default=Path('reanalysis'))
    parser.add_argument('--raw-dir',type=Path)
    parser.add_argument('--bank',type=Path)
    parser.add_argument('--diagnostic-plots',action='store_true',help='Generate legacy diagnostic charts, not the submitted seven plates.')
    args=parser.parse_args();root=Path(__file__).resolve().parent;out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['JMS_ANALYSIS_DIR']=str(out);env['JMS_FIGURE_DIR']=str(out/'figures')
    for name in ['replay_outcomes.csv','reliability_features.npz','bank_eligibility.json','source_manifest.json','spectral_profiles.csv']:
        shutil.copy2(root/'source_data'/name,out/name)
    def run(name,*extra):subprocess.run([sys.executable,str(root/'analysis_code'/name),*extra],env=env,check=True)
    if args.mode=='full':
        if not args.raw_dir:parser.error('full requires --raw-dir containing the four original CSVs or G1/G2 ZIPs and corrected G3/G4 CSVs')
        env['JMS_RAW_DIR']=str(args.raw_dir.resolve());run('prepare_data.py');run('run_stress.py')
    if args.mode=='replay':
        if not args.bank:parser.error('replay requires --bank pointing to the permission-controlled derived replay bank')
        shutil.copy2(args.bank.resolve(),out/'replay_banks.npz');run('run_stress.py','--use-bank')
    run('evaluate_reliability.py')
    if args.mode!='metrics':run('sensitivity_and_checks.py')
    run('supplemental_analysis.py','--predictions',str(out/'heldout_predictions.csv'),'--outdir',str(out/'conditional_alignment'))
    if args.diagnostic_plots:run('make_figures.py')
    print('Completed:',out)
if __name__=='__main__':main()
