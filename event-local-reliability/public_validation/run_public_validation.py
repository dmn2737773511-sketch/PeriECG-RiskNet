"""Execute FROZEN_PLAN.md without outcome-dependent exclusions or tuning."""
from pathlib import Path
import sys, json, csv, hashlib, time, warnings, platform
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, welch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.exceptions import ConvergenceWarning
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
import wfdb
import run_stress_fixed as fixed

FS = 500
N = 10000
S04 = butter(4, [0.5, 40], btype='bandpass', fs=FS, output='sos')
BEATS = {'N','L','R','B','A','a','J','S','V','r','F','e','j','n','E','/','f','Q','?'}
SHIFTS = (0,69,196,367)
RATIOS = (20.,10.,0.,-5.)

def writecsv(name, rows):
    with (ROOT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def readfiltered(dataset, record):
    r = wfdb.rdrecord(str(ROOT/'raw'/dataset/record))
    if float(r.fs) != 360 or r.n_sig != 2:
        raise ValueError(f'Unexpected source dimensions: {record} {r.fs} {r.n_sig}')
    if r.units != ['mV','mV']:
        raise ValueError(f'Unexpected source units: {record}: {r.units}')
    return sosfiltfilt(S04, resample_poly(r.p_signal,25,18,axis=0), axis=0), r

def make_banks():
    targets = []
    clean_rows = []
    symbol_rows = []
    records = (ROOT/'raw'/'mitdb'/'RECORDS').read_text().split()
    for record in records:
        x, info = readfiltered('mitdb', record)
        ann = wfdb.rdann(str(ROOT/'raw'/'mitdb'/record), 'atr')
        keep = np.array([s in BEATS for s in ann.symbol])
        all_locations = np.rint(ann.sample*25/18).astype(np.int64)
        beat_locations = np.rint(ann.sample[keep]*25/18).astype(np.int64)
        for start in (60,120,180,240):
            lo = start*FS
            clean = np.array(x[lo:lo+N],copy=True)
            refs = beat_locations[(beat_locations>=lo)&(beat_locations<lo+N)]-lo
            if len(clean) != N:
                raise ValueError(f'Incomplete target {record} {start}')
            primary = fixed.score(fixed.detect(clean[:,0]),refs)
            amplitude = fixed.score(fixed.detect(clean[:,0],'amplitude'),refs)
            tid = f'mitdb_{record}_s{start}'
            patient = '201_202' if record in ('201','202') else record
            full_counts = Counter(s for s,k in zip(ann.symbol,all_locations) if lo<=k<lo+N)
            scored_counts = Counter(s for s,k in zip(ann.symbol,all_locations) if lo+1000<=k<lo+9000)
            for symbol in sorted(full_counts):
                symbol_rows.append(dict(target=tid,record=record,start_seconds=start,symbol=symbol,
                    included_as_beat_reference=int(symbol in BEATS),
                    count_20s=full_counts[symbol],count_scored_16s=scored_counts[symbol]))
            target = dict(target=tid, record=record, patient_group=patient,
                          start_seconds=start, clean=clean, refs=refs,
                          clean_f1=primary['f1'], clean_f1_amplitude=amplitude['f1'])
            targets.append(target)
            clean_rows.append(dict(target=tid,record=record,patient_group=patient,
                start_seconds=start,lead_1=info.sig_name[0],lead_2=info.sig_name[1],
                n_reference=len(refs[(refs>=1000)&(refs<9000)]),
                clean_f1=primary['f1'],clean_tp=primary['tp'],clean_fp=primary['fp'],clean_fn=primary['fn'],
                clean_f1_amplitude=amplitude['f1'],amplitude_tp=amplitude['tp'],amplitude_fp=amplitude['fp'],amplitude_fn=amplitude['fn'],
                baseline_primary_success=int(primary['f1']>=.95)))
        print(f'built target record {record}',flush=True)
    residuals=[]
    invariant=[]
    for noise in ('bw','ma','em'):
        x, info = readfiltered('nstdb', noise)
        for start in (0,60):
            residual = np.array(x[start*FS:start*FS+N],copy=True)
            residual-=residual.mean(axis=0)
            residual*=np.hanning(N)[:,None]
            residual-=residual.mean(axis=0)
            energy=np.sum(residual**2)
            power=np.abs(np.fft.fft(residual,axis=0))**2
            covariance=np.cov(residual,rowvar=False,bias=True)
            base=dict(energy=energy,power=power,covariance=covariance)
            errs={name:0. for name in base}
            for shift in SHIFTS:
                rotated=np.roll(residual,shift,axis=0)
                vals=dict(energy=np.sum(rotated**2),power=np.abs(np.fft.fft(rotated,axis=0))**2,
                          covariance=np.cov(rotated,rowvar=False,bias=True))
                for name in vals:
                    rel=float(np.max(np.abs(vals[name]-base[name]))/(np.max(np.abs(base[name]))+1e-30))
                    errs[name]=max(errs[name],rel)
            rid=f'nstdb_{noise}_s{start}'
            residuals.append(dict(residual_id=rid,noise_type=noise,noise_start_seconds=start,
                residual=residual/(np.sqrt(np.mean(residual**2))+1e-12)))
            invariant.append(dict(residual_id=rid,**{f'max_relative_{k}_difference':v for k,v in errs.items()}))
    writecsv('clean_targets.csv',clean_rows)
    writecsv('annotation_symbol_counts.csv',symbol_rows)
    writecsv('residual_invariance.csv',invariant)
    max_ref=max(len(t['refs']) for t in targets)
    refs=np.full((len(targets),max_ref),-1,dtype=np.int64)
    for i,t in enumerate(targets): refs[i,:len(t['refs'])]=t['refs']
    np.savez_compressed(ROOT/'public_replay_banks.npz',clean=np.stack([t['clean'] for t in targets]),
        refs=refs,target_ids=np.array([t['target'] for t in targets]),
        records=np.array([t['record'] for t in targets]),
        patient_groups=np.array([t['patient_group'] for t in targets]),
        start_seconds=np.array([t['start_seconds'] for t in targets]),
        clean_f1=np.array([t['clean_f1'] for t in targets]),
        clean_f1_amplitude=np.array([t['clean_f1_amplitude'] for t in targets]),
        residuals=np.stack([n['residual'] for n in residuals]),
        residual_ids=np.array([n['residual_id'] for n in residuals]),
        noise_types=np.array([n['noise_type'] for n in residuals]),
        noise_start_seconds=np.array([n['noise_start_seconds'] for n in residuals]))
    return targets,residuals

def features(x):
    f,p=welch(x,fs=FS,nperseg=1000,noverlap=500,axis=0)
    total=p[(f>=.5)&(f<=40)].sum(axis=0)+1e-12
    ans=[];names=[]
    for j in range(2):
        prefix=f'channel_{j+1}'
        for lo,hi,tag in ((.5,5,'slow'),(5,15,'qrs'),(15,40.01,'fast')):
            ans.append(p[(f>=lo)&(f<hi),j].sum()/total[j])
            names.append(f'{prefix}_{tag}_fraction')
        v=x[:,j];sd=np.std(v)+1e-9
        ans.extend((np.log(sd),np.mean((v-v.mean())**4)/sd**4,np.quantile(np.abs(np.diff(v)),.99)/sd))
        names.extend((f'{prefix}_log_sd',f'{prefix}_kurtosis',f'{prefix}_diff99'))
    ans.append(np.corrcoef(x.T)[0,1]);names.append('corr_channel_1_2')
    primary=fixed.detect(x[:,0]);second=fixed.detect(x[:,1]);amplitude=fixed.detect(x[:,0],'amplitude')
    ans.extend((fixed.agreement(primary,second),fixed.agreement(primary,amplitude)))
    names.extend(('beat_agreement_channel_1_2','detector_agreement_channel_1'))
    for j in range(2):
        snippets=np.array([x[k-60:k+100,j] for k in primary if k>=60 and k+100<len(x)])
        if len(snippets)>2:
            snippets-=np.mean(snippets,axis=1,keepdims=True)
            normalized=snippets/(np.linalg.norm(snippets,axis=1,keepdims=True)+1e-12)
            template=np.median(normalized,axis=0);template/=np.linalg.norm(template)+1e-12
            cosine=normalized@template
            ans.extend((float(np.median(cosine)),float(np.quantile(cosine,.1))))
        else: ans.extend((0.,0.))
        names.extend((f'channel_{j+1}_beat_cos_median',f'channel_{j+1}_beat_cos_q10'))
    return np.nan_to_num(ans),names,primary,amplitude

def replay_target(item):
    t, residuals=item
    clean=t['clean'];scale=np.sqrt(np.mean(clean**2))
    rows=[];values=[]
    for n in residuals:
        for ratio in RATIOS:
            for shift in SHIFTS:
                perturb=np.roll(n['residual'],shift,axis=0)*scale*10**(-ratio/20)
                ft,names,primary,amplitude=features(clean+perturb)
                s=fixed.score(primary,t['refs']);a=fixed.score(amplitude,t['refs'])
                rows.append(dict(target=t['target'],record=t['record'],patient_group=t['patient_group'],
                    start_seconds=t['start_seconds'],residual=n['residual_id'],noise_type=n['noise_type'],
                    noise_start_seconds=n['noise_start_seconds'],snr_db=ratio,shift_samples=shift,
                    clean_f1=t['clean_f1'],clean_f1_amplitude=t['clean_f1_amplitude'],
                    baseline_primary_success=int(t['clean_f1']>=.95),
                    noise_rms_mV=float(np.sqrt(np.mean(perturb**2))),
                    f1=s['f1'],tp=s['tp'],fp=s['fp'],fn=s['fn'],failure=int(s['f1']<.95),
                    f1_amplitude=a['f1'],amplitude_tp=a['tp'],amplitude_fp=a['fp'],amplitude_fn=a['fn']))
                values.append(ft)
    return rows,values,names

def ece(y,p):
    error=0.
    for i in range(10):
        lo=i/10; hi=(i+1)/10
        mask=(p>=lo)&((p<hi) if i<9 else (p<=hi))
        if mask.any():error+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(error)

def metrics(y,p,calibrated=True):
    result=dict(AUROC=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
                AP=float(average_precision_score(y,p)) if y.sum() else None)
    if calibrated:result.update(Brier=float(brier_score_loss(y,p)),ECE_10bin=ece(y,p))
    return result

def evaluate(rows,X,names):
    y=np.array([r['failure'] for r in rows]);f1=np.array([r['f1'] for r in rows])
    groups=np.array([r['patient_group'] for r in rows]);baseline=np.array([r['baseline_primary_success'] for r in rows],dtype=bool)
    models={'Global':np.arange(13),'Event':np.arange(13,19),'Combined':np.arange(19)}
    predictions={name:np.full(len(y),np.nan) for name in models}
    predictions['Agreement']=1-np.minimum(X[:,13],X[:,14])
    cards=[];folds=[];fit_warnings=[]
    for group in sorted(set(groups)):
        train=groups!=group;test=groups==group
        for name,cols in models.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,random_state=20261003))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(X[train][:,cols],y[train])
            for w in caught:fit_warnings.append(dict(patient_group=group,method=name,message=str(w.message)))
            risk=model.predict_proba(X[test][:,cols])[:,1];predictions[name][test]=risk
            scaler=model[0];clf=model[1]
            cards.append(dict(patient_group=group,method=name,n_train=int(train.sum()),n_test=int(test.sum()),
                features=[names[i] for i in cols],mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                coefficients=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0]),n_iter=int(clf.n_iter_[0])))
            folds.append(dict(patient_group=group,method=name,n_train=int(train.sum()),n_test=int(test.sum()),
                              n_failure=int(y[test].sum()),failure_rate=float(y[test].mean()),**metrics(y[test],risk)))
        print(f'LOPO fit {group}',flush=True)
    if any(not np.isfinite(p).all() for p in predictions.values()):raise ValueError('Incomplete OOF predictions')
    summary=[];coverage=[];fixed_retention=[];bins=[]
    for subset,mask in (('all_targets',np.ones(len(y),dtype=bool)),('clean_primary_success',baseline)):
        ys=y[mask];fs=f1[mask]
        for name,pred in predictions.items():
            p=pred[mask]
            summary.append(dict(subset=subset,method=name,n=len(ys),n_failure=int(ys.sum()),failure_rate=float(ys.mean()),
                                **metrics(ys,p,calibrated=name!='Agreement')))
            order=np.argsort(p,kind='stable')
            for cov in (.5,.7,.8,1.):
                ix=order[:int(np.ceil(len(order)*cov))]
                coverage.append(dict(subset=subset,method=name,coverage=cov,n_total=len(ys),n_retained=len(ix),
                    n_retained_failure=int(ys[ix].sum()),failure_rate_retained=float(ys[ix].mean()),mean_f1_retained=float(fs[ix].mean())))
            if name!='Agreement':
                retained=p<=.1
                fixed_retention.append(dict(subset=subset,method=name,risk_threshold=.1,n_total=len(ys),n_retained=int(retained.sum()),
                    coverage=float(retained.mean()),n_retained_failure=int(ys[retained].sum()),
                    failure_rate_retained=float(ys[retained].mean()) if retained.any() else None,
                    mean_f1_retained=float(fs[retained].mean()) if retained.any() else None))
                for i in range(10):
                    m=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
                    bins.append(dict(subset=subset,method=name,bin_lower=i/10,bin_upper=(i+1)/10,n=int(m.sum()),
                        observed_failure=float(ys[m].mean()) if m.any() else None,mean_risk=float(p[m].mean()) if m.any() else None))
    quartet_rows=[]
    for start in range(0,len(rows),4):
        ix=np.arange(start,start+4);r=rows[start];k=int(y[ix].sum());q=k/4
        row=dict(target=r['target'],record=r['record'],patient_group=r['patient_group'],residual=r['residual'],
            noise_type=r['noise_type'],snr_db=r['snr_db'],baseline_primary_success=r['baseline_primary_success'],
            n_failure=k,mixed_failure=int(0<k<4),min_f1=float(f1[ix].min()),max_f1=float(f1[ix].max()),
            f1_range=float(np.ptp(f1[ix])),minimum_invariant_errors=min(k,4-k),minimum_invariant_brier=q*(1-q),
            n_success_failure_pairs=k*(4-k))
        for name,p in predictions.items():
            good=ix[y[ix]==0];bad=ix[y[ix]==1]
            diff=p[bad,None]-p[good][None,:]
            row[f'pair_concordance_{name}']=float(np.mean((diff>0)+.5*(diff==0))) if diff.size else None
            row[f'pair_correct_credit_{name}']=float(np.sum((diff>0)+.5*(diff==0)))
        quartet_rows.append(row)
    quartet_summary=[]
    for subset in ('all_targets','clean_primary_success'):
        for ratio in ('all',20.,10.,0.,-5.):
            selected=[r for r in quartet_rows if (subset=='all_targets' or r['baseline_primary_success']) and (ratio=='all' or r['snr_db']==ratio)]
            qcount=len(selected);pair_count=sum(r['n_success_failure_pairs'] for r in selected)
            row=dict(subset=subset,snr_db=ratio,n_quartets=qcount,n_episodes=qcount*4,
                n_mixed_quartets=sum(r['mixed_failure'] for r in selected),
                mixed_fraction=sum(r['mixed_failure'] for r in selected)/qcount,
                minimum_invariant_errors=sum(r['minimum_invariant_errors'] for r in selected),
                minimum_invariant_error_rate=sum(r['minimum_invariant_errors'] for r in selected)/(qcount*4),
                minimum_invariant_brier=float(np.mean([r['minimum_invariant_brier'] for r in selected])),n_success_failure_pairs=pair_count)
            for name in predictions:
                row[f'pair_concordance_{name}']=sum(r[f'pair_correct_credit_{name}'] for r in selected)/pair_count if pair_count else None
                mixed=[r[f'pair_concordance_{name}'] for r in selected if r['mixed_failure']]
                row[f'quartet_equal_concordance_{name}']=float(np.mean(mixed)) if mixed else None
            quartet_summary.append(row)
    out=[]
    for i,row in enumerate(rows):out.append(dict(**row,**{f'risk_{k}':float(v[i]) for k,v in predictions.items()}))
    writecsv('heldout_predictions.csv',out)
    writecsv('model_summary.csv',summary)
    writecsv('patient_fold_metrics.csv',folds)
    writecsv('risk_coverage.csv',coverage)
    writecsv('fixed_threshold_retention.csv',fixed_retention)
    writecsv('calibration_bins.csv',bins)
    writecsv('matched_quartets.csv',quartet_rows)
    writecsv('quartet_summary.csv',quartet_summary)
    (ROOT/'fitted_models.json').write_text(json.dumps(cards,indent=2))
    (ROOT/'fit_warnings.json').write_text(json.dumps(fit_warnings,indent=2))
    snr_rows=[]
    for subset,mask in (('all_targets',np.ones(len(y),dtype=bool)),('clean_primary_success',baseline)):
        for noise in ('all','bw','ma','em'):
            for ratio in ('all',20.,10.,0.,-5.):
                m=mask&np.array([noise=='all' or r['noise_type']==noise for r in rows])&np.array([ratio=='all' or r['snr_db']==ratio for r in rows])
                snr_rows.append(dict(subset=subset,noise_type=noise,snr_db=ratio,n=int(m.sum()),n_failure=int(y[m].sum()),
                    failure_rate=float(y[m].mean()),mean_f1=float(f1[m].mean()),
                    mean_f1_amplitude=float(np.array([r['f1_amplitude'] for r in rows])[m].mean())))
    writecsv('outcome_summary.csv',snr_rows)
    return dict(n_records=48,n_patient_groups=len(set(groups)),n_targets=192,n_noise_windows=6,n_episodes=len(rows),
        n_quartets=len(quartet_rows),n_clean_primary_success=int(baseline.sum()/96),
        mean_f1=float(f1.mean()),n_failures=int(y.sum()),failure_rate=float(y.mean()),
        model_summary=summary,quartet_summary=quartet_summary,fixed_threshold_retention=fixed_retention,
        convergence_warnings=fit_warnings)

def main():
    started=time.time()
    targets,residuals=make_banks()
    all_rows=[];all_values=[];names=None
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,(rows,values,names) in enumerate(pool.map(replay_target,[(t,residuals) for t in targets])):
            all_rows.extend(rows);all_values.extend(values)
            if i%8==0:print(f'replayed {i+1}/192 targets; elapsed {time.time()-started:.1f}s',flush=True)
    X=np.array(all_values)
    writecsv('replay_outcomes.csv',all_rows)
    np.savez_compressed(ROOT/'reliability_features.npz',X=X,names=np.array(names),global_n=13)
    result=evaluate(all_rows,X,names)
    result.update(elapsed_seconds=time.time()-started,python=platform.python_version(),wfdb=wfdb.__version__,
        fixed_detector_sha256=hashlib.sha256((ROOT/'run_stress_fixed.py').read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256((ROOT/'FROZEN_PLAN.md').read_bytes()).hexdigest())
    (ROOT/'analysis_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
