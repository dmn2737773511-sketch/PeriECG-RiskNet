import os, sys
from pathlib import Path
import csv,json,sys,platform
import numpy as np,scipy,sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path(os.environ.get('JMS_ANALYSIS_DIR', str(Path(__file__).resolve().parents[1]/'source_data')))
def rd(p):return list(csv.DictReader(open(p)))
def wr(p,r):
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0]);w.writeheader();w.writerows(r)
r=rd(ROOT/'replay_outcomes.csv');z=np.load(ROOT/'reliability_features.npz');X=z['X'];gn=int(z['global_n']);tg=np.array([a['target_group'] for a in r]);ng=np.array([a['residual_group'] for a in r]);y0=np.array([float(a['f1']) for a in r]);ya=np.array([float(a['f1_amplitude']) for a in r]);cases=[]
for detector,vals,tolerances in [('energy',y0,[.9,.95,.98]),('amplitude',ya,[.95])]:
 for threshold in tolerances:
  y=(vals<threshold).astype(int)
  methods={'Global':np.arange(gn),'Event':np.arange(gn,X.shape[1])}
  if detector=='energy' and threshold==.95:methods.update({'Agreement_features':np.arange(gn,gn+3),'Morphology_features':np.arange(gn+3,X.shape[1])})
  for key,cols in methods.items():
   p=np.full(len(y),np.nan)
   for g in ['G1','G2','G3','G4']:
    tr=(tg!=g)&(ng!=g);te=(tg==g)&(ng==g)
    mod=make_pipeline(StandardScaler(),LogisticRegression(C=1,solver='lbfgs',max_iter=2000,random_state=20261003));mod.fit(X[tr][:,cols],y[tr]);p[te]=mod.predict_proba(X[te][:,cols])[:,1]
   ok=np.isfinite(p);yp=y[ok];pp=p[ok];order=np.argsort(pp,kind='stable');idx=order[:int(np.ceil(.8*len(order)))];cases.append(dict(detector=detector,failure_threshold=threshold,method=key,n=int(ok.sum()),AUROC=float(roc_auc_score(yp,pp)),AUPRC=float(average_precision_score(yp,pp)),Brier=float(brier_score_loss(yp,pp)),risk_at_80pct=float(yp[idx].mean())))
wr(ROOT/'sensitivity_analyses.csv',cases)
# Verify rotation invariants of every residual bank.
import run_stress as rs
_targets,_noise=rs.load_banks(ROOT/'replay_banks.npz');banks={'targets':_targets,'noise':_noise};max_fft=max_energy=max_cov=0.
for no in banks['noise']:
 a=no['residual'];f=np.fft.rfft(a,axis=0);p=np.abs(f)**2;cov=a.T@a
 for k in [0,69,196,367]:
  b=np.roll(a,k,axis=0);q=np.abs(np.fft.rfft(b,axis=0))**2
  max_fft=max(max_fft,float(np.max(np.abs(q-p))/(np.max(p)+1e-15)))
  max_energy=max(max_energy,float(abs(np.mean(a*a)-np.mean(b*b))/(np.mean(a*a)+1e-15)))
  max_cov=max(max_cov,float(np.max(np.abs(b.T@b-cov))/(np.max(np.abs(cov))+1e-15)))
# Seven displayed leads remain an exact transformation of three acquired leads.
A=np.array([[1,0,0],[0,1,0],[-1,1,0],[-.5,-.5,0],[1,-.5,0],[-.5,1,0],[0,0,1]])
source_overlap = 0
for g in ['G1','G2','G3','G4']:
 tr=(tg!=g)&(ng!=g);te=(tg==g)&(ng==g)
 train_targets={r[i]['target'] for i in np.flatnonzero(tr)}; test_targets={r[i]['target'] for i in np.flatnonzero(te)}
 train_residuals={r[i]['residual'] for i in np.flatnonzero(tr)}; test_residuals={r[i]['residual'] for i in np.flatnonzero(te)}
 source_overlap += len(train_targets & test_targets) + len(train_residuals & test_residuals)
 assert tr.sum()==9216 and te.sum()==1024
assert source_overlap == 0
checks=dict(rotation_max_relative_periodogram_error=max_fft,rotation_max_relative_energy_error=max_energy,rotation_max_relative_covariance_error=max_cov,lead_map_rank=int(np.linalg.matrix_rank(A)),n_source_recordings=4,heldout_source_overlap=int(source_overlap),clean_eligibility_all_pass=all(a.get('eligible',False) for a in json.load(open(ROOT/'bank_eligibility.json')) if a['kind']=='target'),software=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,scikit_learn=sklearn.__version__,platform=platform.platform()))
assert max_fft<1e-12 and max_energy<1e-12 and max_cov<1e-12
# Recompute sixteen saved replay rows independently from bank definitions and compare features.
import run_stress as rs
lookup_t={a['id']:a for a in banks['targets']};lookup_n={a['id']:a for a in banks['noise']};max_feature=0
for i in np.linspace(0,len(r)-1,16,dtype=int):
 row=r[i];t=lookup_t[row['target']];no=lookup_n[row['residual']];res=no['residual'];res=res/np.sqrt(np.mean(res**2)+0) # epsilon below mirrors the calculation
 res=no['residual']/(np.sqrt(np.mean(no['residual']**2))+1e-12)
 perturb=np.roll(res,int(row['shift_samples']),axis=0)*np.sqrt(np.mean(t['clean']**2))*10**(-float(row['snr_db'])/20)
 ft,names,gn,p,pa=rs.features(t['clean']+perturb)
 max_feature=max(max_feature,float(np.max(np.abs(ft-X[i]))));assert abs(rs.score(p,t['events'])['f1']-float(row['f1']))<1e-12
checks['replayed_saved_rows']=16;checks['max_feature_recompute_abs_error']=max_feature;assert max_feature<1e-10
json.dump(checks,open(ROOT/'verification_report.json','w'),indent=2);print(json.dumps(checks,indent=2));print(cases)
