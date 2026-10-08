#!/usr/bin/env python3
"""Regenerate the seven main and two supplementary plates from their numeric sources."""
from pathlib import Path
import argparse,os,subprocess,sys

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-cache',required=True,type=Path,help='Directory with G1_processed.npz through G4_processed.npz, produced by full analysis.')
    p.add_argument('--bank',type=Path,default=root/'author_only/replay_banks.npz',help='Authorized waveform-derived bank.')
    p.add_argument('--data-dir',type=Path,default=root/'source_data',help='Saved numerical evaluation tables.')
    p.add_argument('--out',type=Path,default=root/'figures',help='Output directory for assembled plates and individual panels.')
    a=p.parse_args()
    for f in [a.bank,*[a.raw_cache/f'G{i}_processed.npz' for i in range(1,5)]]:
        if not f.is_file():p.error(f'Missing input: {f}. See README for raw-data reconstruction and permissions.')
    env=os.environ.copy()
    for k,v in {'JMS_PACKAGE_DIR':root,'JMS_TABLE_DIR':a.data_dir,'JMS_RAW_CACHE':a.raw_cache,'JMS_REPLAY_BANK':a.bank,'JMS_FIGURE_OUTPUT':a.out}.items():env[k]=str(v.resolve())
    env.setdefault('OPENBLAS_NUM_THREADS','1')
    for name in ['derive_figure_tables.py','build_panels.py','build_supplement_panels.py']:
        subprocess.run([sys.executable,str(root/'analysis_code'/name)],env=env,check=True)
    print('Completed:',a.out.resolve())
if __name__=='__main__':main()
