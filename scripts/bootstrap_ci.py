#!/usr/bin/env python3
"""Compute bootstrap 95% CI for patient-level metrics from a patient_probs.csv file.

Usage:
  python scripts/bootstrap_ci.py --probs outputs/exp4-attention/patient_probs.csv --n_boot 1000 --out outputs/exp4-attention/bootstrap_ci.json
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json


def compute_metrics(y, p):
    # find Youden threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    best = thr[np.argmax(j)]
    y_pred = (p >= best).astype(int)
    acc = accuracy_score(y, y_pred)
    pre = precision_score(y, y_pred, zero_division=0)
    sen = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y, p)
    return {'auc': float(auc), 'acc': float(acc), 'pre': float(pre), 'sen': float(sen), 'f1': float(f1), 'spe': float(spe)}


def bootstrap_ci(df, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    y = df['label'].values
    p = df['prob'].values
    stats = {k: [] for k in ['auc','acc','pre','sen','f1','spe']}
    for i in range(n_boot):
        idx = rng.randint(0, len(y), size=len(y))
        yb = y[idx]
        pb = p[idx]
        m = compute_metrics(yb, pb)
        for k in stats.keys():
            stats[k].append(m[k])
    ci = {}
    for k, arr in stats.items():
        a = np.array(arr)
        ci[k] = {'median': float(np.median(a)), '2.5': float(np.percentile(a, 2.5)), '97.5': float(np.percentile(a, 97.5))}
    # point estimate on full sample
    pe = compute_metrics(y, p)
    return {'point_estimate': pe, 'bootstrap_ci': ci}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probs', required=True, help='CSV with columns patient_id,label,prob')
    parser.add_argument('--n_boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.probs)
    if 'prob' not in df.columns:
        raise SystemExit('probs csv must contain column named prob')
    res = bootstrap_ci(df, n_boot=args.n_boot, seed=args.seed)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print('Saved bootstrap CI to', args.out)


if __name__ == '__main__':
    main()
