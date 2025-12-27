#!/usr/bin/env python3
"""Cross-validation helper for patient-level evaluation.

Usage examples:
  # inference-only (fast): uses an existing base checkpoint to infer on each fold's val set
  python scripts/cross_validate.py --manifest data/train_manifest.csv --config outputs/exp4-attention/config.yaml \
    --base_ckpt outputs/exp4-attention/final_best.pth --n_splits 5 --out_dir outputs/cv --mode inference_only

  # full-train (will call train.py for each fold)  -- slower
  python scripts/cross_validate.py --manifest data/train_manifest.csv --config config.yaml --n_splits 5 --out_dir outputs/cv --mode full_train
"""

import argparse
import os
import yaml
import shutil
import sys
# ensure repo root is on sys.path so local imports like `data_utils` work when
# running the script from within `scripts/` or other locations
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import subprocess
import time
from data_utils import PDVolDataset, PatchDataset
from models_patch import build_patch_model, PatchMILAggregator
from torch.utils.data import DataLoader


def eval_fold(cfg, val_manifest, base_ckpt, out_fold):
    os.makedirs(out_fold, exist_ok=True)
    # load cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    model = build_patch_model(cfg).to(device)
    ckpt = torch.load(base_ckpt, map_location=device)
    if all(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.eval()

    # prepare val dataset
    val_dataset = PDVolDataset(manifest_csv=val_manifest, input_size=cfg['data']['input_size'], mode='val')
    patch_val = PatchDataset(base_dataset=val_dataset, patch_size=tuple(cfg['train']['patch_size']), stride=tuple(cfg['train']['patch_stride']), mode='pretrain')
    loader = DataLoader(patch_val, batch_size=cfg['train'].get('batch_size', 64), shuffle=False, num_workers=cfg['train'].get('num_workers', 4))

    pooling_type = cfg['train'].get('pooling', 'topk_mean')
    if pooling_type != 'attention' and os.path.exists(os.path.join(out_fold, '..', '..', 'aggregator_best.pth')):
        pooling_type = 'attention'
    need_features = pooling_type in ('attention', 'distribution', 'soft_topk', 'quantile')

    patient_results = {}
    with torch.no_grad():
        for batch in loader:
            x = batch['volume'].to(device)
            pids = batch['patient_id']
            labels = batch['label']
            if need_features:
                logits, feats = model(x, return_features=True)
                logits = logits.cpu().numpy()
                feats = feats.cpu().numpy()
            else:
                logits = model(x).cpu().numpy()
                feats = [None] * len(logits)
            for i, pid in enumerate(pids):
                if pid not in patient_results:
                    patient_results[pid] = {'logits': [], 'features': [], 'label': int(labels[i])}
                patient_results[pid]['logits'].append(logits[i])
                if need_features:
                    patient_results[pid]['features'].append(feats[i])

    # aggregator
    feat_dim = 0
    if need_features and len(patient_results) > 0:
        first = list(patient_results.keys())[0]
        if len(patient_results[first]['features']) > 0 and patient_results[first]['features'][0] is not None:
            feat_dim = np.array(patient_results[first]['features'][0]).shape[0]

    aggregator = PatchMILAggregator(feat_dim=feat_dim, pooling=pooling_type, topk_percent=cfg['train'].get('topk_percent', 0.15)).to(device)
    agg_ckpt = os.path.join(out_fold, '..', '..', 'aggregator_best.pth')
    if pooling_type == 'attention' and os.path.exists(agg_ckpt):
        try:
            ag = torch.load(agg_ckpt, map_location=device)
            if all(k.startswith('module.') for k in ag.keys()):
                ag = {k.replace('module.', ''): v for k, v in ag.items()}
            aggregator.load_state_dict(ag)
        except Exception:
            pass

    y_true = []
    y_prob = []
    for pid, data in patient_results.items():
        logits = torch.tensor(data['logits']).to(device).unsqueeze(0)
        feats = torch.tensor(np.array(data['features'])).to(device).unsqueeze(0) if need_features else None
        patient_logit = aggregator(logits, feats)
        prob = torch.sigmoid(patient_logit).item()
        y_true.append(data['label'])
        y_prob.append(prob)

    # save probs
    df = pd.DataFrame({'patient_id': list(patient_results.keys()), 'label': y_true, 'prob': y_prob})
    df.to_csv(os.path.join(out_fold, 'patient_probs.csv'), index=False)

    # metrics
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = np.argmax(j)
    best_th = thresholds[best_idx]
    y_pred = (y_prob >= best_th).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob)
    metrics = {'auc': float(auc), 'acc': float(acc), 'sen': float(sen), 'pre': float(pre), 'f1': float(f1), 'spe': float(spe), 'best_th': float(best_th)}
    with open(os.path.join(out_fold, 'metrics.yaml'), 'w') as f:
        yaml.safe_dump(metrics, f)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--mode', choices=['inference_only', 'full_train'], default='inference_only')
    parser.add_argument('--base_ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, names=['qsm','t1','aal','label','id']) if pd.read_csv(args.manifest, nrows=1, header=None).shape[1]==5 else pd.read_csv(args.manifest)
    # ensure columns
    if 'id' not in df.columns:
        # try last column as id
        df.columns = ['qsm','t1','aal','label','id']

    patients = df[['id','label']].drop_duplicates().reset_index(drop=True)
    X = patients['id'].values
    y = patients['label'].values

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    fold_metrics = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        val_ids = set(X[val_idx])
        fold_dir = os.path.join(args.out_dir, f'fold_{i}')
        os.makedirs(fold_dir, exist_ok=True)
        val_manifest = os.path.join(fold_dir, 'val_manifest.csv')
        df[df['id'].isin(val_ids)].to_csv(val_manifest, index=False, header=False)

        # write per-fold train manifest as well (for full_train mode)
        train_manifest = os.path.join(fold_dir, 'train_manifest.csv')
        df[~df['id'].isin(val_ids)].to_csv(train_manifest, index=False, header=False)

        # prepare fold-specific config path (default to global config)
        fold_cfg_path = args.config

        if args.mode == 'full_train':
            # create a per-fold config that points to the fold-specific train/val manifests
            try:
                base_cfg = yaml.safe_load(open(args.config))
            except Exception:
                base_cfg = {}
            base_cfg.setdefault('data', {})
            base_cfg['data']['train_csv'] = train_manifest
            base_cfg['data']['val_csv'] = val_manifest
            # enforce attention-based PatchMIL (stage5) for fold training
            base_cfg.setdefault('train', {})
            base_cfg['train']['pooling'] = 'attention'
            fold_cfg_path = os.path.join(fold_dir, 'config.yaml')
            with open(fold_cfg_path, 'w') as _f:
                yaml.safe_dump(base_cfg, _f)

            # call train.py for this fold using the fold-specific config
            out_fold_train = os.path.join(fold_dir, 'train_out')
            # use the same Python executable running this script to spawn train.py subprocesses
            cmd_list = [sys.executable, os.path.join(repo_root, 'train.py'), '--config', fold_cfg_path, '--out_dir', out_fold_train]
            print('Running full training for fold', i, 'cmd=', ' '.join(cmd_list))
            os.makedirs(out_fold_train, exist_ok=True)
            # attempt training with retries in case of OOM: decrease batch_size/num_workers and retry
            max_retries = 3
            attempt = 0
            trained_successfully = False
            base_ckpt = os.path.join(out_fold_train, 'final_best.pth')
            while attempt < max_retries and not trained_successfully:
                attempt += 1
                log_path = os.path.join(out_fold_train, f'train_run_attempt{attempt}.log')
                with open(log_path, 'wb') as logf:
                    proc = subprocess.run(cmd_list, stdout=logf, stderr=subprocess.STDOUT)
                if os.path.exists(base_ckpt):
                    trained_successfully = True
                    print(f"Fold {i} training succeeded on attempt {attempt}")
                    break
                # not successful: inspect log for OOM or other errors
                try:
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        log_text = f.read()
                except Exception:
                    log_text = ''
                if 'OutOfMemoryError' in log_text or 'CUDA out of memory' in log_text or proc.returncode != 0:
                    print(f"Fold {i} training failed on attempt {attempt} (OOM or error). Returncode={proc.returncode}.")
                    # try to reduce batch_size / num_workers in fold config before retrying
                    try:
                        fc = yaml.safe_load(open(fold_cfg_path))
                    except Exception:
                        fc = {}
                    train_cfg = fc.setdefault('train', {})
                    old_bs = train_cfg.get('batch_size', None)
                    if old_bs is None:
                        # try to read from global config if available
                        old_bs = train_cfg.get('batch_size', 64)
                    new_bs = max(1, int(old_bs // 2)) if old_bs else 1
                    train_cfg['batch_size'] = new_bs
                    # cap num_workers to small number
                    train_cfg['num_workers'] = min(4, max(1, int(train_cfg.get('num_workers', 4)//2)))
                    with open(fold_cfg_path, 'w') as _f:
                        yaml.safe_dump(fc, _f)
                    print(f"Retrying with reduced batch_size={train_cfg['batch_size']} num_workers={train_cfg['num_workers']}")
                    # small pause before retry
                    time.sleep(5)
                    continue
                else:
                    # unknown failure but checkpoint missing; break to avoid infinite loop
                    print(f"Fold {i} training finished without producing checkpoint and without OOM (attempt {attempt}). Check log: {log_path}")
                    break
            if not trained_successfully:
                print(f"Fold {i}: training failed after {attempt} attempts, skipping evaluation for this fold.")
        else:
            base_ckpt = args.base_ckpt

        # load config for evaluation (use fold config if created)
        eval_cfg = yaml.safe_load(open(fold_cfg_path)) if fold_cfg_path and os.path.exists(fold_cfg_path) else yaml.safe_load(open(args.config))
        # only evaluate if checkpoint exists
        if base_ckpt is None or not os.path.exists(base_ckpt):
            print(f"No checkpoint found for fold {i} at {base_ckpt}; writing empty metrics and continuing.")
            metrics = { 'auc': float('nan'), 'acc': float('nan'), 'sen': float('nan'), 'pre': float('nan'), 'f1': float('nan'), 'spe': float('nan'), 'best_th': float('nan') }
            with open(os.path.join(fold_dir, 'metrics.yaml'), 'w') as f:
                yaml.safe_dump(metrics, f)
        else:
            metrics = eval_fold(eval_cfg, val_manifest, base_ckpt, fold_dir)
        print('Fold', i, 'metrics:', metrics)
        fold_metrics.append(metrics)

    # summarize
    keys = fold_metrics[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        summary[k] = {'mean': float(vals.mean()), 'std': float(vals.std())}
    with open(os.path.join(args.out_dir, 'cv_summary.yaml'), 'w') as f:
        yaml.safe_dump(summary, f)
    print('CV summary saved to', os.path.join(args.out_dir, 'cv_summary.yaml'))


if __name__ == '__main__':
    main()
