#!/usr/bin/env python3
"""Paired alignment discrimination and fixed-threshold decisions.

Input is authentic source-excluded prediction output. No fitting is performed.
Quartet-equal concordance is primary; pair-weighted concordance is secondary.
Agreement is an uncalibrated ranking score, so its thresholds are score thresholds.
No episode-independent inference, p values, or population confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

MODELS = ('Global', 'Event', 'Combined', 'Agreement')
THRESHOLDS = (0.05, 0.10, 0.20)
EXPECTED_SHIFTS = (0, 69, 196, 367)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No output rows for {path}')
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_predictions(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    required = {'target', 'residual', 'snr_db', 'shift_samples', 'target_group',
                'residual_group', 'failure', 'f1', 'tp', 'fp', 'fn'}
    required.update('risk_' + m for m in MODELS)
    if not rows or required.difference(rows[0]):
        raise ValueError(f'Missing prediction columns: {required.difference(rows[0] if rows else {})}')
    seen = set()
    for row in rows:
        for key in ('shift_samples', 'failure', 'tp', 'fp', 'fn'):
            row[key] = int(row[key])
        for key in ('snr_db', 'f1') + tuple('risk_' + m for m in MODELS):
            row[key] = float(row[key])
        key = (row['target'], row['residual'], row['snr_db'], row['shift_samples'])
        if key in seen:
            raise ValueError(f'Duplicate episode ID: {key}')
        seen.add(key)
        if row['target_group'] != row['residual_group']:
            raise ValueError('Input is not the diagonal source-excluded test set')
        if row['failure'] != int(row['f1'] < 0.95):
            raise ValueError(f'Failure label mismatch: {key}')
        if row['shift_samples'] not in EXPECTED_SHIFTS:
            raise ValueError(f'Unexpected rotation: {key}')
        if not 0.0 <= row['f1'] <= 1.0:
            raise ValueError(f'Invalid F1: {key}')
        for m in MODELS:
            p = row['risk_' + m]
            if not 0.0 <= p <= 1.0:
                raise ValueError(f'Nonfinite or out-of-range score for {m}: {key}')
    return rows


def strata(rows):
    groups = [('pooled', 'all', rows)]
    for source in sorted({r['target_group'] for r in rows}):
        groups.append(('source', source, [r for r in rows if r['target_group'] == source]))
    for snr in sorted({r['snr_db'] for r in rows}, reverse=True):
        groups.append(('snr', f'{snr:g}', [r for r in rows if r['snr_db'] == snr]))
    for source in sorted({r['target_group'] for r in rows}):
        for snr in sorted({r['snr_db'] for r in rows}, reverse=True):
            groups.append(('source_snr', f'{source}:{snr:g}',
                           [r for r in rows if r['target_group'] == source and r['snr_db'] == snr]))
    return groups


def analyze_quartets(rows):
    quartets = defaultdict(list)
    for row in rows:
        quartets[(row['target'], row['residual'], row['snr_db'])].append(row)
    details = []
    for (target, residual, snr), q in sorted(quartets.items()):
        if len(q) != 4 or tuple(sorted(r['shift_samples'] for r in q)) != EXPECTED_SHIFTS:
            raise ValueError(f'Incomplete quartet: {(target, residual, snr)}')
        sources = {r['target_group'] for r in q}
        if len(sources) != 1:
            raise ValueError(f'Inconsistent quartet source: {(target, residual, snr)}')
        failures = [r for r in q if r['failure']]
        nonfailures = [r for r in q if not r['failure']]
        pairs = len(failures) * len(nonfailures)
        for model in MODELS:
            win = tie = loss = 0
            for a in failures:
                for b in nonfailures:
                    delta = a['risk_' + model] - b['risk_' + model]
                    if delta > 0:
                        win += 1
                    elif delta == 0:
                        tie += 1
                    else:
                        loss += 1
            scores = [r['risk_' + model] for r in q]
            details.append({
                'target': target, 'residual': residual, 'snr_db': snr,
                'source': next(iter(sources)), 'model': model,
                'quartet_failure_count': len(failures), 'mixed': int(pairs > 0),
                'failure_nonfailure_pairs': pairs, 'concordant_pairs': win,
                'tied_pairs': tie, 'discordant_pairs': loss,
                'quartet_concordance': (win + 0.5 * tie) / pairs if pairs else None,
                'score_min': min(scores), 'score_max': max(scores),
                'score_range': max(scores) - min(scores),
                'f1_min': min(r['f1'] for r in q), 'f1_max': max(r['f1'] for r in q),
            })
    summary = []
    for scope, key, selected in strata(rows):
        selected_keys = {(r['target'], r['residual'], r['snr_db']) for r in selected}
        for model in MODELS:
            all_q = [d for d in details if d['model'] == model and
                     (d['target'], d['residual'], d['snr_db']) in selected_keys]
            mixed = [d for d in all_q if d['mixed']]
            pair_count = sum(d['failure_nonfailure_pairs'] for d in mixed)
            wins = sum(d['concordant_pairs'] for d in mixed)
            ties = sum(d['tied_pairs'] for d in mixed)
            summary.append({
                'scope': scope, 'stratum': key, 'model': model,
                'total_quartets': len(all_q), 'mixed_quartets': len(mixed),
                'failure_nonfailure_pairs': pair_count,
                'concordant_pairs': wins, 'tied_pairs': ties,
                'discordant_pairs': sum(d['discordant_pairs'] for d in mixed),
                'concordance_quartet_equal_primary': mean(d['quartet_concordance'] for d in mixed) if mixed else None,
                'concordance_pair_weighted_secondary': (wins + 0.5 * ties) / pair_count if pair_count else None,
            })
    return details, summary


def analyze_thresholds(rows):
    decisions = []
    for scope, key, selected in strata(rows):
        n = len(selected)
        failures_total = sum(r['failure'] for r in selected)
        nonfailures_total = n - failures_total
        for model in MODELS:
            for threshold in THRESHOLDS:
                retained = [r for r in selected if r['risk_' + model] <= threshold]
                k = len(retained)
                failure_count = sum(r['failure'] for r in retained)
                tp = sum(r['tp'] for r in retained)
                fp = sum(r['fp'] for r in retained)
                fn = sum(r['fn'] for r in retained)
                discarded_nonfailures = nonfailures_total - (k - failure_count)
                decisions.append({
                    'scope': scope, 'stratum': key, 'model': model,
                    'threshold_kind': 'uncalibrated_score' if model == 'Agreement' else 'predicted_probability',
                    'threshold': threshold, 'comparison': '<=',
                    'episodes': n, 'all_failures': failures_total,
                    'all_nonfailures': nonfailures_total, 'retained': k,
                    'coverage': k / n if n else None,
                    'retained_failures': failure_count,
                    'retained_failure_fraction': failure_count / k if k else None,
                    'discarded_nonfailures': discarded_nonfailures,
                    'discarded_nonfailure_fraction': discarded_nonfailures / nonfailures_total if nonfailures_total else None,
                    'retained_mean_f1': mean(r['f1'] for r in retained) if k else None,
                    'retained_tp': tp, 'retained_fp': fp, 'retained_fn': fn,
                    'retained_beat_precision': tp / (tp + fp) if tp + fp else None,
                    'retained_beat_recall': tp / (tp + fn) if tp + fn else None,
                    'retained_micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                    'oracle_failure_floor_at_retained_count': max(0, k - nonfailures_total) / k if k else None,
                })
    return decisions


def run(input_path, output_dir):
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_predictions(input_path)
    details, concordance = analyze_quartets(rows)
    thresholds = analyze_thresholds(rows)
    for scope, key, _ in strata(rows):
        for model in MODELS:
            d = [r for r in thresholds if r['scope'] == scope and r['stratum'] == key and r['model'] == model]
            if [r['retained'] for r in d] != sorted(r['retained'] for r in d):
                raise ValueError('Fixed threshold retained counts are not nested')
            if [r['retained_failures'] for r in d] != sorted(r['retained_failures'] for r in d):
                raise ValueError('Fixed threshold retained failures are not nested')
            for r in d:
                if r['all_nonfailures'] - (r['retained'] - r['retained_failures']) != r['discarded_nonfailures']:
                    raise ValueError('Nonfailure retention accounting failed')
    output_tables = {
        'quartet_pair_details.csv': details,
        'quartet_concordance_summary.csv': concordance,
        'fixed_threshold_decisions.csv': [r for r in thresholds if r['scope'] in ('pooled', 'source')],
        'fixed_threshold_by_snr.csv': [r for r in thresholds if r['scope'] in ('snr', 'source_snr')],
    }
    for name, table in output_tables.items():
        write_csv(output_dir / name, table)
    summary = {
        'input_sha256': sha256(input_path),
        'episode_count': len(rows),
        'source_groups': sorted({r['target_group'] for r in rows}),
        'heldout_failures': sum(r['failure'] for r in rows),
        'quartet_concordance_pooled': [r for r in concordance if r['scope'] == 'pooled'],
        'quartet_concordance_by_source': [r for r in concordance if r['scope'] == 'source'],
        'quartet_concordance_by_snr': [r for r in concordance if r['scope'] == 'snr'],
        'fixed_threshold_decisions_pooled': [r for r in thresholds if r['scope'] == 'pooled'],
        'fixed_threshold_decisions_by_source': [r for r in thresholds if r['scope'] == 'source'],
        'interpretation': {
            'primary': 'Average concordance across mixed held-out quartets, with quartets weighted equally.',
            'secondary': 'Concordance across all failure/nonfailure pairs in mixed quartets, with pairs weighted equally.',
            'ties': 'Exact equal scores receive one-half credit.',
            'fixed_thresholds': 'Thresholds 0.05, 0.10, and 0.20 are fixed descriptive research operating points; no test-set optimization.',
            'agreement': 'Agreement is an uncalibrated ranking score; its threshold is not a probability threshold.',
            'scope': 'Recording-source-excluded computational replay from one coded series; no independent-patient inference.',
        },
    }
    (output_dir / 'supplemental_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    provenance = {
        'executed_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(),
        'script_sha256': sha256(__file__),
        'input': str(input_path),
        'input_sha256': sha256(input_path),
        'checks': {
            'episode_id_unique': True,
            'all_input_sources_diagonal_heldout_pairs': True,
            'all_quartets_complete_four_expected_rotations': True,
            'failure_labels_equal_f1_below_0_95': True,
            'scores_finite_in_unit_interval': True,
            'threshold_retained_counts_and_failure_counts_nested': True,
            'discarded_nonfailure_accounting_exact': True,
            'model_fitting_performed': False,
            'iid_inference_performed': False,
        },
        'outputs': {name: {'rows': len(table), 'sha256': sha256(output_dir / name)} for name, table in output_tables.items()},
    }
    (output_dir / 'supplemental_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default=str(Path(__file__).resolve().parents[1] / 'source_data' / 'heldout_predictions.csv'))
    parser.add_argument('--outdir', default=str(Path(__file__).resolve().parents[1] / 'source_data' / 'conditional_alignment'))
    args = parser.parse_args()
    summary = run(args.predictions, args.outdir)
    print(json.dumps({
        'episodes': summary['episode_count'],
        'failures': summary['heldout_failures'],
        'quartet_concordance_pooled': summary['quartet_concordance_pooled'],
        'output_directory': str(Path(args.outdir).resolve()),
    }, indent=2))
