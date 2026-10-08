"""Descriptive reporting supplement; no detectors, features or models rerun."""
from pathlib import Path
from collections import Counter
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(name):return list(csv.DictReader((ROOT/name).open()))
def write(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    quartets=read('matched_quartets.csv')
    summary=read('quartet_summary.csv')
    methods=('Global','Event','Combined','Agreement')
    for row in summary:
        selected=[r for r in quartets if (row['subset']=='all_targets' or int(r['baseline_primary_success'])) and
            (row['snr_db']=='all' or float(r['snr_db'])==float(row['snr_db']))]
        for method in methods:
            values=[float(r[f'pair_concordance_{method}']) for r in selected if int(r['mixed_failure'])]
            row[f'quartet_equal_concordance_{method}']=float(np.mean(values)) if values else None
    write('quartet_summary.csv',summary)
    analysis=json.loads((ROOT/'analysis_summary.json').read_text())
    for row in analysis['quartet_summary']:
        matching=next(r for r in summary if r['subset']==row['subset'] and str(r['snr_db'])==str(row['snr_db']))
        for method in methods:row[f'quartet_equal_concordance_{method}']=matching[f'quartet_equal_concordance_{method}']
    analysis['reporting_addendum']='REPORTING_ADDENDUM.md; descriptive summaries only; no model or design adjustment'
    (ROOT/'analysis_summary.json').write_text(json.dumps(analysis,indent=2))

    clean=read('clean_targets.csv');outcomes=read('replay_outcomes.csv');folds=read('patient_fold_metrics.csv')
    patients=sorted({r['patient_group'] for r in outcomes})
    baseline={}
    for key in ('clean_f1','clean_f1_amplitude'):
        f1=np.array([float(r[key]) for r in clean])
        baseline[key]=dict(n=len(f1),mean=float(f1.mean()),median=float(np.median(f1)),minimum=float(f1.min()),maximum=float(f1.max()),
            n_failure=int((f1<.95).sum()),n_success=int((f1>=.95).sum()),n_perfect=int((f1==1).sum()))
    counts=Counter()
    counts_scored=Counter()
    counts_ref=Counter()
    for r in read('annotation_symbol_counts.csv'):
        counts[r['symbol']]+=int(r['count_20s']);counts_scored[r['symbol']]+=int(r['count_scored_16s'])
        if int(r['included_as_beat_reference']):counts_ref[r['symbol']]+=int(r['count_scored_16s'])
    with np.load(ROOT/'reliability_features.npz') as f:
        dimension=f['X'].shape;names=list(map(str,f['names']));finite=bool(np.isfinite(f['X']).all())
    manifest=json.loads((ROOT/'download_manifest.json').read_text())
    checks=dict(n_records=len({r['record'] for r in clean}),n_patients=len(patients),n_targets=len(clean),n_episodes=len(outcomes),
        n_quartets=len(quartets),n_download_files=len(manifest),total_download_bytes=sum(r['bytes'] for r in manifest),
        clean_baseline=baseline,annotation_counts_20s=dict(counts),annotation_counts_scored_16s=dict(counts_scored),
        reference_beat_counts_scored_16s=dict(counts_ref),n_reference_beats_scored_16s=sum(counts_ref.values()),
        n_learning_symbols_scored=counts_scored['?'],n_zero_reference_targets=sum(int(r['n_reference'])==0 for r in clean),
        feature_shape=list(dimension),feature_names=names,all_features_finite=finite,
        excluded_information_in_features=sorted(set(names)&{'reference','record','patient_group','target','residual','snr_db','clean_f1'}),
        clean_201_202_groups={r['record']:r['patient_group'] for r in clean if r['record'] in ('201','202')},
        test_rows_201_202=sum(r['patient_group']=='201_202' for r in outcomes))
    (ROOT/'descriptive_verification.json').write_text(json.dumps(checks,indent=2))
    patient_summary=[]
    for method in ('Global','Event','Combined'):
        selected=[r for r in folds if r['method']==method]
        for metric in ('AUROC','AP','Brier','ECE_10bin'):
            values=np.array([float(r[metric]) for r in selected if r[metric]])
            patient_summary.append(dict(method=method,metric=metric,n_patient_groups=len(selected),n_defined=len(values),
                patient_equal_mean=float(values.mean()),patient_equal_median=float(np.median(values)),
                minimum=float(values.min()),maximum=float(values.max())))
    write('patient_equal_metric_summary.csv',patient_summary)
    holdout=[]
    for patient in patients:
        train=[r for r in outcomes if r['patient_group']!=patient]
        test=[r for r in outcomes if r['patient_group']==patient]
        train_patients={r['patient_group'] for r in train};test_patients={r['patient_group'] for r in test}
        train_noise={r['residual'] for r in train};test_noise={r['residual'] for r in test}
        holdout.append(dict(heldout_patient_group=patient,n_train=len(train),n_test=len(test),
            n_train_patient_groups=len(train_patients),test_records=sorted({r['record'] for r in test}),
            train_test_patient_overlap=sorted(train_patients&test_patients),
            n_shared_noise_windows=len(train_noise&test_noise),noise_windows_shared=True,
            evaluation_scope='unseen target patient; shared fixed benchmark noise bank'))
    (ROOT/'source_holdout_checks.json').write_text(json.dumps(holdout,indent=2))
    predictions=read('heldout_predictions.csv')
    policy_rows=[];policy_summary=[]
    for method in ('Global','Event','Combined'):
        for patient in patients:
            selected=[r for r in predictions if r['patient_group']==patient]
            retained=[r for r in selected if float(r[f'risk_{method}'])<=.1]
            excluded=[r for r in selected if float(r[f'risk_{method}'])>.1]
            nf=sum(int(r['failure']) for r in selected)
            nr=sum(int(r['failure']) for r in retained)
            nb=sum(int(r['failure']) and not int(r['baseline_primary_success']) for r in retained)
            nc=sum(int(r['failure']) and int(r['baseline_primary_success']) for r in retained)
            policy_rows.append(dict(method=method,patient_group=patient,risk_threshold=.1,n_total=len(selected),
                n_failure=nf,n_retained=len(retained),coverage=len(retained)/len(selected),n_retained_failure=nr,
                failure_rate_retained=nr/len(retained) if retained else None,
                n_retained_failure_clean_baseline_failed=nb,n_retained_failure_clean_baseline_success=nc,
                n_excluded=len(excluded),n_excluded_failure=sum(int(r['failure']) for r in excluded),
                n_excluded_nonfailure=sum(not int(r['failure']) for r in excluded)))
        for subset in ('all_targets','clean_primary_success','clean_primary_failure'):
            selected=[r for r in predictions if subset=='all_targets' or
                (subset=='clean_primary_success' and int(r['baseline_primary_success'])) or
                (subset=='clean_primary_failure' and not int(r['baseline_primary_success']))]
            retained=[r for r in selected if float(r[f'risk_{method}'])<=.1]
            excluded=[r for r in selected if float(r[f'risk_{method}'])>.1]
            row=dict(method=method,subset=subset,risk_threshold=.1,n_total=len(selected),
                n_total_failure=sum(int(r['failure']) for r in selected),
                n_retained=len(retained),n_retained_failure=sum(int(r['failure']) for r in retained),
                n_retained_failure_clean_baseline_failed=sum(int(r['failure']) and not int(r['baseline_primary_success']) for r in retained),
                n_retained_failure_clean_baseline_success=sum(int(r['failure']) and int(r['baseline_primary_success']) for r in retained),
                n_retained_nonfailure=sum(not int(r['failure']) for r in retained),
                n_excluded=len(excluded),n_excluded_failure=sum(int(r['failure']) for r in excluded),
                n_excluded_nonfailure=sum(not int(r['failure']) for r in excluded))
            assert row['n_total']==row['n_retained']+row['n_excluded']
            assert row['n_total_failure']==row['n_retained_failure']+row['n_excluded_failure']
            assert row['n_retained_failure']==row['n_retained_failure_clean_baseline_failed']+row['n_retained_failure_clean_baseline_success']
            if subset=='all_targets':
                patient_policies=[r for r in policy_rows if r['method']==method]
                row['n_patient_groups_zero_retained']=sum(r['n_retained']==0 for r in patient_policies)
                row['maximum_patient_failure_rate_retained']=max(r['failure_rate_retained'] for r in patient_policies if r['n_retained'])
                row['maximum_failure_patient_groups']=[r['patient_group'] for r in patient_policies if r['failure_rate_retained']==row['maximum_patient_failure_rate_retained']]
            policy_summary.append(row)
    write('fixed_threshold_by_patient.csv',policy_rows)
    (ROOT/'fixed_threshold_source_counts.json').write_text(json.dumps(policy_summary,indent=2))
    print(json.dumps(dict(baseline=baseline,patient_AUROC=[r for r in patient_summary if r['metric']=='AUROC'],checks={k:checks[k] for k in ('n_reference_beats_scored_16s','n_learning_symbols_scored','n_zero_reference_targets','clean_201_202_groups','test_rows_201_202')}),indent=2))

if __name__=='__main__':main()
