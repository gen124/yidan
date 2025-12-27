#!/usr/bin/env python3
"""
Evaluate an experiment using distribution pooling.

Computes AUC, ACC, SEN, SPE, PRE, F1 at patient level.
Usage:
  python scripts/eval_distribution.py --exp_dir outputs/exp_distribution --config outputs/exp_distribution/config.yaml
Optional: --limit N to run on only N patients (for quick smoke test)
"""
import os
import sys
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import csv

# ensure project root is on PYTHONPATH when running from scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_utils import PDVolDataset, PatchDataset
from models_patch import build_patch_model, PatchMILAggregator


def eval_distribution(exp_dir, config_path, batch_size=64, num_workers=4, limit=None):
    # load config (prefer exp_dir/config.yaml)
    exp_cfg = os.path.join(exp_dir, 'config.yaml')
    if os.path.exists(exp_cfg):
        config_path = exp_cfg

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device: {device}")

    # load model
    model = build_patch_model(cfg).to(device)
    ckpt_path = os.path.join(exp_dir, 'final_best.pth')
    if not os.path.exists(ckpt_path):
        # fallback to pretrain_best or relabel_iter2
        for alt in ['pretrain_best.pth', 'relabel_iter2.pth', 'relabel_iter1.pth']:
            altp = os.path.join(exp_dir, alt)
            if os.path.exists(altp):
                ckpt_path = altp
                print(f"[WARN] final_best.pth not found, falling back to {alt}")
                break
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in {exp_dir}")

    ck = torch.load(ckpt_path, map_location=device)
    if isinstance(ck, dict) and 'state_dict' in ck:
        state_dict = ck['state_dict']
    else:
        state_dict = ck
    # strip module. prefix if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load val dataset
    val_dataset = PDVolDataset(manifest_csv=cfg['data']['val_csv'], input_size=cfg['data']['input_size'], mode='val')
    print(f"[INFO] val patients: {len(val_dataset)}")

    patch_size = tuple(cfg['train']['patch_size'])
    patch_stride = tuple(cfg['train']['patch_stride'])

    # build patch dataset covering full validation set
    patch_val_dataset = PatchDataset(base_dataset=val_dataset, patch_size=patch_size, stride=patch_stride, mode='pretrain')
    print(f"[INFO] total patches in val: {len(patch_val_dataset)}")

    loader = DataLoader(patch_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # collect per-patient logits and features
    patient_results = {}

    with torch.no_grad():
        for batch in loader:
            x = batch['volume'].to(device)
            pids = batch['patient_id']
            labels = batch['label']
            # forward
            logits, features = model(x, return_features=True)
            logits = logits.cpu().numpy()
            features = features.cpu().numpy()
            for i, pid in enumerate(pids):
                if limit is not None and len(patient_results) >= limit:
                    break
                if pid not in patient_results:
                    patient_results[pid] = {'logits': [], 'features': [], 'label': int(labels[i])}
                patient_results[pid]['logits'].append(float(logits[i]))
                patient_results[pid]['features'].append(features[i])
            if limit is not None and len(patient_results) >= limit:
                break

    # determine feat_dim
    feat_dim = 0
    if len(patient_results) > 0:
        first = next(iter(patient_results.values()))
        if len(first['features']) > 0:
            feat_dim = int(np.array(first['features'][0]).shape[0])

    # build aggregator: force distribution and try to load trained aggregator if available
    pooling_type = cfg['train'].get('pooling', 'distribution')
    aggregator = PatchMILAggregator(feat_dim=feat_dim, pooling=pooling_type, topk_percent=cfg['train'].get('topk_percent', 0.15)).to(device)

    # try to load aggregator checkpoint saved during Stage 5 (name includes pooling)
    def _load_agg_ckpt(aggregator, exp_dir, device):
        candidates = [os.path.join(exp_dir, f"aggregator_{pooling_type}_best.pth"), os.path.join(exp_dir, 'aggregator_best.pth')]
        for p in candidates:
            if os.path.exists(p):
                try:
                    ck = torch.load(p, map_location=device)
                    if isinstance(ck, dict) and 'state_dict' in ck:
                        sd = ck['state_dict']
                    else:
                        sd = ck
                    # strip module. prefix if present
                    if any(k.startswith('module.') for k in sd.keys()):
                        sd = {k.replace('module.', ''): v for k, v in sd.items()}
                    aggregator.load_state_dict(sd)
                    print(f"[INFO] loaded aggregator checkpoint: {p}")
                    return True
                except Exception as e:
                    print(f"[WARN] failed loading aggregator checkpoint {p}: {e}")
        print(f"[WARN] no aggregator checkpoint found for pooling='{pooling_type}', using freshly initialized aggregator")
        return False

    _load_agg_ckpt(aggregator, exp_dir, device)

    y_true = []
    y_prob = []
    patient_ids = []

    for pid, data in patient_results.items():
        logits_arr = np.array(data['logits'])
        feats_arr = np.array(data['features'])
        logits_t = torch.tensor(logits_arr).to(device).unsqueeze(0)  # (1, N)
        feats_t = torch.tensor(feats_arr).to(device).unsqueeze(0)    # (1, N, D)
        with torch.no_grad():
            patient_logit = aggregator(logits_t, feats_t)
            prob = float(torch.sigmoid(patient_logit).item())
        y_true.append(data['label'])
        y_prob.append(prob)
        patient_ids.append(pid)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = np.argmax(j)
    youden_threshold = float(thresholds[best_idx])

    # Best F1 threshold: search on dense grid
    thr_grid = np.linspace(0.0, 1.0, 1001)
    best_f1 = -1.0
    best_f1_thr = 0.5
    for thr in thr_grid:
        y_pred_thr = (y_prob >= thr).astype(int)
        f1t = f1_score(y_true, y_pred_thr, zero_division=0)
        if f1t > best_f1:
            best_f1 = float(f1t)
            best_f1_thr = float(thr)

    # helper to compute per-threshold metrics
    def _metrics_at_threshold(threshold):
        y_pred_thr = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_thr, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc_t = float((tp + tn) / max(1, (tp + tn + fp + fn)))
        prec_t = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall_t = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_t = float((2 * prec_t * recall_t) / (prec_t + recall_t)) if (prec_t + recall_t) > 0 else 0.0
        spec_t = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return {
            'threshold': float(threshold),
            'accuracy': acc_t,
            'precision': prec_t,
            'recall': recall_t,
            'f1': f1_t,
            'specificity': spec_t,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }

    # compute metrics for default 0.5, youden, best_f1
    metrics_default = _metrics_at_threshold(0.5)
    metrics_youden = _metrics_at_threshold(youden_threshold)
    metrics_bestf1 = _metrics_at_threshold(best_f1_thr)

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan')
    brier = float(np.mean((y_prob - y_true) ** 2))

    out = {
        'auc': auc,
        'youden_threshold': youden_threshold,
        'best_f1_threshold': best_f1_thr,
        'per_threshold': {
            'default_0.5': metrics_default,
            'youden': metrics_youden,
            'best_f1': metrics_bestf1
        },
        'brier': brier
    }

    # print summary
    print("\n" + "="*40)
    print(f"Evaluation (distribution) - {exp_dir}")
    print(f"Patients: {len(y_true)}, Pos: {int(y_true.sum())}, Neg: {len(y_true)-int(y_true.sum())}")
    print(f"AUC: {auc:.6f}")
    print(f"Youden threshold: {youden_threshold:.6f}")
    print(f"Best-F1 threshold: {best_f1_thr:.6f}")
    print(f"Brier score: {brier:.6f}")
    print("\nPer-threshold metrics:")
    for k, v in out['per_threshold'].items():
        print(f" - {k}: threshold={v['threshold']}, acc={v['accuracy']:.4f}, prec={v['precision']:.4f}, recall={v['recall']:.4f}, f1={v['f1']:.4f}, spec={v['specificity']:.4f}, tn={v['tn']}, fp={v['fp']}, fn={v['fn']}, tp={v['tp']}")
    print("="*40)

    # save JSON to exp_dir
    out_path = os.path.join(exp_dir, 'eval_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] saved metrics to {out_path}")

    # save per-patient predictions using three thresholds: default 0.5, youden, best_f1
    preds_csv = os.path.join(exp_dir, 'patient_predictions.csv')
    with open(preds_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'label', 'prob', 'pred_0.5', 'pred_youden', 'pred_best_f1'])
        for pid, lbl, prob in zip(patient_ids, y_true.tolist(), y_prob.tolist()):
            p0 = int(prob >= 0.5)
            py = int(prob >= youden_threshold)
            pb = int(prob >= best_f1_thr)
            writer.writerow([pid, int(lbl), float(prob), p0, py, pb])
    print(f"[INFO] saved per-patient predictions to {preds_csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', required=True)
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=None, help='Limit number of patients for quick test')
    args = p.parse_args()
    eval_distribution(args.exp_dir, args.config, batch_size=args.batch_size, num_workers=args.num_workers, limit=args.limit)
