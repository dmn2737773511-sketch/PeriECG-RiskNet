import os, sys
from pathlib import Path
import csv,json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve
ROOT=Path(os.environ.get('JMS_ANALYSIS_DIR', str(Path(__file__).resolve().parents[1]/'source_data')))
def readcsv(p):return list(csv.DictReader(open(p)))
def writecsv(p,rows):
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def ece(y,p):
 e=0.
 for a in np.linspace(0,.9,10):
  m=(p>=a)&((p<a+.1) if a<.9 else (p<=1))
  if m.any():e+=m.mean()*abs(y[m].mean()-p[m].mean())
 return float(e)
def metr(y,p):return dict(AUROC=float(roc_auc_score(y,p)),AUPRC=float(average_precision_score(y,p)),Brier=float(brier_score_loss(y,p)),ECE_10bin=ece(y,p))
r=readcsv(ROOT/'replay_outcomes.csv');z=np.load(ROOT/'reliability_features.npz');X=z['X'];gn=int(z['global_n']);names=list(z['names'])
y=np.array([int(a['failure']) for a in r]);f1=np.array([float(a['f1']) for a in r]);fa=np.array([float(a['f1_amplitude']) for a in r]); tg=np.array([a['target_group'] for a in r]);ng=np.array([a['residual_group'] for a in r]); pattern=np.array([a['pattern'] for a in r]);snrs=np.array([float(a['snr_db']) for a in r])
models={'Global':np.arange(gn),'Event':np.arange(gn,X.shape[1]),'Combined':np.arange(X.shape[1])}
prs={k:np.full(len(y),np.nan) for k in [*models,'Agreement']};foldmetrics=[];coefficients=[];cards=[]
for g in ['G1','G2','G3','G4']:
 train=(tg!=g)&(ng!=g);test=(tg==g)&(ng==g)
 for key,cols in models.items():
  model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,random_state=20261003))
  model.fit(X[train][:,cols],y[train]);p=model.predict_proba(X[test][:,cols])[:,1];prs[key][test]=p
  foldmetrics.append(dict(heldout_record=g,method=key,n_train=int(train.sum()),n_test=int(test.sum()),failure_rate=float(y[test].mean()),**metr(y[test],p)))
  scale=model[0];clf=model[1]
  cards.append(dict(record=g,method=key,features=[str(names[i]) for i in cols],mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coefficients=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0])))
 prs['Agreement'][test]=1-np.minimum.reduce([X[test,gn],X[test,gn+1],X[test,gn+2]])
sel=np.isfinite(prs['Global']);ys=y[sel];combined=[];curves=[];patternrows=[]
for key in prs:
 p=prs[key][sel];m=metr(ys,p);m.update(method=key,n=int(sel.sum()))
 order=np.argsort(p,kind='stable')
 for cov in [0.25,.5,.6,.7,.8,.9,1.]:
  ix=order[:int(np.ceil(len(order)*cov))];risk=float(ys[ix].mean());rr=dict(method=key,coverage=cov,n_retained=len(ix),failure_rate=risk,mean_f1=float(f1[sel][ix].mean()))
  curves.append(rr)
  if cov==.8:m['risk_at_80pct']=risk;m['mean_f1_at_80pct']=rr['mean_f1']
 for pat in ['regular','variable']:
  ix=order[:int(np.ceil(len(order)*.8))];mask=(pattern[sel]==pat);kept=(pattern[sel][ix]==pat)
  patternrows.append(dict(method=key,pattern=pat,n_total=int(mask.sum()),n_retained=int(kept.sum()),retention_rate=float(kept.sum()/mask.sum()),failure_rate_retained=float(ys[ix][kept].mean())))
 combined.append(m)
writecsv(ROOT/'reliability_summary.csv',combined);writecsv(ROOT/'fold_metrics.csv',foldmetrics);writecsv(ROOT/'risk_coverage.csv',curves);writecsv(ROOT/'pattern_retention.csv',patternrows);json.dump(cards,open(ROOT/'fitted_reliability_models.json','w'),indent=2)
pred=[]
for i,a in enumerate(r):
 if not sel[i]:continue
 pred.append(dict(**a,**{f'risk_{k}':float(v[i]) for k,v in prs.items()}))
writecsv(ROOT/'heldout_predictions.csv',pred)
quartets={}
for i,a in enumerate(r):
 key=(a['target'],a['residual'],a['snr_db']);quartets.setdefault(key,[]).append(i)
qr=[]
for key,ix in quartets.items():
 vals=f1[ix];aa=fa[ix]
 qr.append(dict(target=key[0],residual=key[1],snr_db=key[2],min_f1=float(vals.min()),max_f1=float(vals.max()),f1_range=float(vals.max()-vals.min()),straddles_failure=int(vals.min()<.95 and vals.max()>=.95),amplitude_f1_range=float(aa.max()-aa.min()),noise_rms_relative_range=float(np.ptp([float(r[i]['noise_rms_mV']) for i in ix])/float(r[ix[0]]['noise_rms_mV']))))
writecsv(ROOT/'matched_alignment_quartets.csv',qr)
summary=dict(n_templates=32,n_clean_episodes=64,n_residuals=16,n_replay=len(r),n_heldout=int(sel.sum()),global_features=gn,total_features=X.shape[1],n_quartets=len(qr),n_straddling=sum(q['straddles_failure'] for q in qr),straddling_fraction=np.mean([q['straddles_failure'] for q in qr]),max_f1_range=max(q['f1_range'] for q in qr),median_f1_range=np.median([q['f1_range'] for q in qr]),max_relative_noise_rms_difference=max(q['noise_rms_relative_range'] for q in qr),overall_mean_f1=float(f1.mean()),overall_failure_rate=float(y.mean()),heldout_failure_rate=float(ys.mean()),model_summary=combined)
ss=[]
for snr in [20.,10.,0.,-5.]:
 ix=snrs==snr;qq=[q for q in qr if float(q['snr_db'])==snr]
 ss.append(dict(snr_db=snr,n=int(ix.sum()),mean_f1=float(f1[ix].mean()),median_f1=float(np.median(f1[ix])),failure_rate=float(y[ix].mean()),mean_f1_amplitude=float(fa[ix].mean()),fraction_straddling=np.mean([q['straddles_failure'] for q in qq]),mean_alignment_f1_range=np.mean([q['f1_range'] for q in qq])))
writecsv(ROOT/'snr_summary.csv',ss);summary['snr_summary']=ss
json.dump(summary,open(ROOT/'analysis_summary.json','w'),indent=2)
print(json.dumps(summary,indent=2))
