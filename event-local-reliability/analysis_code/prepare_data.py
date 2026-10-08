import os
from pathlib import Path
import csv,json,zipfile,hashlib
import numpy as np
from scipy.signal import butter,sosfiltfilt,welch,find_peaks
OUT=Path(os.environ['JMS_ANALYSIS_DIR']); OUT.mkdir(exist_ok=True,parents=True)
base=Path(os.environ['JMS_RAW_DIR'])
for p in base.glob('S01*.zip'):
 with zipfile.ZipFile(p) as z:
  for info in z.infolist():
   if info.filename.endswith('.csv'):
    dst=OUT/Path(info.filename).name
    if not dst.exists():dst.write_bytes(z.read(info.filename))
candidates=sorted(OUT.glob('S01*.csv'))+sorted(base.glob('S01*.csv'))
paths=[]
for tag in ['G1_noHydrogel_garment','G2_hydrogel_garment','G3_silverIon_noHydrogel','G4_hydrogelCouplant_garment']:
 matches=[p for p in candidates if tag in p.name and not p.name.endswith('_events.csv')]
 if tag.startswith(('G3','G4')): matches=[p for p in matches if 'corrected' in p.name]
 if not matches: raise FileNotFoundError(f'Missing source CSV for {tag}')
 paths.append(matches[0])
leads=['I_V','II_V','III_V','aVR_V','aVL_V','aVF_V','V1_V','V2_V','V3_V','V4_V','V5_V','V6_V']
manifest=[]; spec=[]
for path in paths:
 with path.open(encoding='utf-8-sig') as f:
  meta={};nhead=0
  for line in f:
   if not line.startswith('#'):
    cols=next(csv.reader([line]));break
   nhead+=1;k,_,v=line[1:].strip().partition(',');meta[k]=v
 g=meta['Group'][:2]; cache=OUT/f'{g}.npz'
 selected=['phase_index','sample_index','nominal_time_s',*leads]
 if cache.exists():a=np.load(cache)['a']
 else:
  a=np.loadtxt(path,delimiter=',',skiprows=nhead+1,usecols=[cols.index(c) for c in selected]);np.savez_compressed(cache,a=a)
 phase=a[:,0].astype(int); x=a[:,3:]*1000;fs=500.
 dif=np.diff(a[:,1]);manifest.append(dict(group=g,source=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),subject=meta.get('SubjectID'),n_samples=len(a),fs=fs,nonunit_sample_steps=int((dif!=1).sum()),phase_counts={str(i):int((phase==i).sum()) for i in range(6)},metadata=meta))
 xf=sosfiltfilt(butter(4,[.5,40],btype='bandpass',fs=fs,output='sos'),x,axis=0)
 for phaseval in range(6):
  for kind,xx in [('raw',x),('filtered',xf)]:
   v=xx[phase==phaseval]
   f,p=welch(v,fs=fs,nperseg=10000,noverlap=5000,axis=0,window='hann',detrend='constant')
   for li,lead in enumerate(leads):
    row=dict(group=g,phase=phaseval,preprocessing=kind,lead=lead[:-2])
    for lo,hi,name in [(.05,.5,'low'),(.5,5,'slow'),(5,15,'qrs'),(15,40,'fast'),(49,51,'line')]:row[name+'_rms_uV']=float(np.sqrt(p[(f>=lo-1e-9)&(f<hi-1e-9),li].sum()*(f[1]-f[0]))*1000)
    spec.append(row)
 np.savez_compressed(OUT/f'{g}_processed.npz',x=x,xf=xf,phase=phase)
 print(g,len(a),manifest[-1]['nonunit_sample_steps'],flush=True)
json.dump(manifest,open(OUT/'source_manifest.json','w'),indent=2,ensure_ascii=False)
with open(OUT/'spectral_profiles.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=spec[0]);w.writeheader();w.writerows(spec)
