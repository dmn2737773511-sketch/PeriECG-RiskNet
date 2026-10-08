from pathlib import Path
import json,sys,os
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
O=Path(os.environ.get('JMS_PACKAGE_DIR',str(Path(__file__).resolve().parents[1])))
D=Path(os.environ.get('JMS_TABLE_DIR',str(O/'source_data')))
BANK=Path(os.environ.get('JMS_REPLAY_BANK',str(O/'author_only/replay_banks.npz')))
r=pd.read_csv(D/'replay_outcomes.csv'); h=pd.read_csv(D/'heldout_predictions.csv');q=pd.read_csv(D/'matched_alignment_quartets.csv')
q['target_group']=q.target.str[:2];q['residual_group']=q.residual.str[:2];q['activity_phase']=q.residual.str[-1].astype(int)
q.to_csv(D/'quartet_details.csv',index=False)
a=r.groupby(['target','residual','snr_db']).failure.agg(['sum','size']).reset_index()
a['minimum_mistakes']=np.minimum(a['sum'],a['size']-a['sum']);a['prevalence']=a['sum']/a['size'];a['minimum_mean_squared_error']=a.prevalence*(1-a.prevalence)
a.to_csv(D/'rotation_invariant_error_bounds.csv',index=False)
bound=[]
for snr in [20,10,0,-5]:
 s=a[a.snr_db==snr];bound.append(dict(snr_db=snr,n_episodes=int(s['size'].sum()),minimum_mistakes=int(s.minimum_mistakes.sum()),minimum_error_rate=s.minimum_mistakes.sum()/s['size'].sum(),minimum_Brier=s.minimum_mean_squared_error.mean()))
bound.append(dict(snr_db='all',n_episodes=len(r),minimum_mistakes=int(a.minimum_mistakes.sum()),minimum_error_rate=a.minimum_mistakes.sum()/len(r),minimum_Brier=a.minimum_mean_squared_error.mean()))
pd.DataFrame(bound).to_csv(D/'rotation_invariant_bound_summary.csv',index=False)
# Full risk/coverage grid and source-specific grids: no model refitting or test-set tuning.
rows=[]
for subset,s in [('pooled',h)]+[(g,h[h.target_group==g]) for g in ['G1','G2','G3','G4']]:
 for model in ['Global','Event','Combined','Agreement']:
  ss=s.sort_values('risk_'+model,kind='stable'); good=(ss.failure==0).sum()
  for c in np.arange(25,101)/100:
   k=int(np.ceil(c*len(ss))); u=ss.iloc[:k]
   rows.append(dict(subset=subset,method=model,coverage=c,n=len(ss),retained=k,retained_failures=int(u.failure.sum()),failure_rate=u.failure.mean(),discarded=len(ss)-k,discarded_nonfailures=int(good-(u.failure==0).sum()),mean_f1=u.f1.mean(),oracle_minimum=max(0,k-good)/k))
pd.DataFrame(rows).to_csv(D/'retention_full_grid.csv',index=False)
# Discrimination conditional on generated timing pattern.
rows=[]
for pattern in ['regular','variable']:
 s=h[h.pattern==pattern];y=s.failure.values
 for model in ['Global','Event','Combined']:
  p=s['risk_'+model].values
  e=0
  for j in range(10):
   m=(p>=j/10)&((p<(j+1)/10) if j<9 else (p<=1))
   if m.any():e+=m.mean()*abs(p[m].mean()-y[m].mean())
  rows.append(dict(pattern=pattern,method=model,n=len(s),failure_rate=y.mean(),AUROC=roc_auc_score(y,p),AP=average_precision_score(y,p),Brier=brier_score_loss(y,p),ECE=e))
pd.DataFrame(rows).to_csv(D/'pattern_metrics.csv',index=False)
# Invariance audit across all sixteen residuals, using only explicit numeric operations.
z=np.load(BANK,allow_pickle=False);inv=[]
for i,rid in enumerate(z['residual_ids']):
 x=z['residuals'][i];p=np.abs(np.fft.rfft(x,axis=0))**2;C=x.T@x
 for shift in [0,69,196,367]:
  u=np.roll(x,shift,axis=0)
  inv.append(dict(residual=str(rid),shift_samples=shift,periodogram_error=np.max(np.abs(np.abs(np.fft.rfft(u,axis=0))**2-p))/(np.max(p)+1e-15),energy_error=abs(np.mean(u*u)-np.mean(x*x))/(np.mean(x*x)+1e-15),covariance_error=np.max(np.abs(u.T@u-C))/(np.max(np.abs(C))+1e-15)))
pd.DataFrame(inv).to_csv(D/'rotation_invariance_all_residuals.csv',index=False)
# Deterministic illustrative selection: largest crossing range among within-source quartets.
u=q[(q.target_group==q.residual_group)&(q.straddles_failure==1)].sort_values(['f1_range','target','residual','snr_db'],ascending=[False,True,True,True],kind='stable').iloc[0]
e=r[(r.target==u.target)&(r.residual==u.residual)&(r.snr_db==u.snr_db)].sort_values('shift_samples')
print('Chosen quartet:',e.to_string(index=False))
order=h.sort_values('risk_Event',kind='stable');k=int(np.ceil(.7*len(order)));kept=order.iloc[:k];removed=order.iloc[k:]
case_low=kept[kept.failure==1].sort_values(['f1','risk_Event']).iloc[0]
outside=~((removed.target==u.target)&(removed.residual==u.residual)&(removed.snr_db==u.snr_db))
case_high=removed[(removed.failure==0)&outside].sort_values('risk_Event',ascending=False).iloc[0]
selection={'matched_quartet':e.to_dict('records'),'retained_failure':case_low.to_dict(),'discarded_nonfailure':case_high.to_dict(),'selection_rules':{'matched_quartet':'Largest F1 range among within-source quartets that cross F1=0.95; illustrative, not representative prevalence.','retained_failure':'Lowest F1 among failures retained by Event at pooled 70% coverage.','discarded_nonfailure':'Highest Event risk among nonfailures excluded at pooled 70% coverage, excluding the Fig. 3 quartet to prevent duplicate illustrations.'}}
json.dump(selection,open(D/'illustrative_case_selection.json','w'),indent=2)
# Verify headline values and source separation from saved data.
assert len(r)==16384 and len(h)==4096 and h.failure.sum()==1043
assert q.straddles_failure.sum()==460
assert int(a.minimum_mistakes.sum())==588
assert int(kept.failure.sum())==44
check={'replay_rows':len(r),'heldout_rows':len(h),'heldout_failures':int(h.failure.sum()),'crossing_quartets':int(q.straddles_failure.sum()),'minimum_invariant_mistakes':int(a.minimum_mistakes.sum()),'minimum_invariant_error':float(a.minimum_mistakes.sum()/len(r)),'rotation_invariant_Brier_floor':float(a.minimum_mean_squared_error.mean()),'event_failures_at70':int(kept.failure.sum()),'revision_scope':'Figure expansion, manuscript rewrite and deterministic analyses of existing outputs. No new human recordings or clinical labels.'}
json.dump(check,open(D/'revision_numeric_checks.json','w'),indent=2)
print('Bounds',bound)
print('Cases\n',case_low[['target','residual','snr_db','shift_samples','f1','risk_Event']],case_high[['target','residual','snr_db','shift_samples','f1','risk_Event']])
