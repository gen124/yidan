import os
import sys
import yaml
import torch
import numpy as np
import json
from pathlib import Path

# ensure project root is on sys.path when run from monitor/nohup
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_utils import PDVolDataset, PatchDataset
from models_patch import build_patch_model, PatchMILAggregator
from tqdm import tqdm


def eval_exp(exp_dir, cfg_path=None, device=None):
    exp_config = os.path.join(exp_dir, 'config.yaml')
    if os.path.exists(exp_config):
        cfg_path = exp_config
    if cfg_path is None:
        cfg_path = 'config.yaml'
    cfg = yaml.safe_load(open(cfg_path))
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model = build_patch_model(cfg).to(device)
    ckpt_path = os.path.join(exp_dir, 'final_best.pth')
    if not os.path.exists(ckpt_path):
        fallback = os.path.join('outputs', 'exp4', 'final_best.pth')
        if os.path.exists(fallback):
            ckpt_path = fallback
        else:
            raise FileNotFoundError(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device)
    if all(k.startswith('module.') for k in ck.keys()):
        ck = {k.replace('module.', ''): v for k, v in ck.items()}
    model.load_state_dict(ck)
    model.eval()

    val_dataset = PDVolDataset(manifest_csv=cfg['data']['val_csv'], input_size=cfg['data']['input_size'], mode='val')
    patch_val_dataset = PatchDataset(base_dataset=val_dataset, patch_size=tuple(cfg['train']['patch_size']), stride=tuple(cfg['train']['patch_stride']), mode='pretrain')
    loader = DataLoader(patch_val_dataset, batch_size=64, shuffle=False, num_workers=4)

    pooling_type = cfg['train'].get('pooling', 'topk_mean')
    if pooling_type != 'attention' and os.path.exists(os.path.join(exp_dir, 'aggregator_best.pth')):
        pooling_type = 'attention'
    need_features = (pooling_type == 'attention')

    patient_results = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Inferring {exp_dir}'):
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

    feat_dim = 0
    if need_features and len(patient_results) > 0:
        first = next(iter(patient_results.values()))
        if len(first['features']) > 0:
            feat_dim = first['features'][0].shape[0]

    aggregator = PatchMILAggregator(feat_dim=feat_dim, pooling=pooling_type, topk_percent=cfg['train'].get('topk_percent', 0.15)).to(device)
    agg_ckpt = os.path.join(exp_dir, 'aggregator_best.pth')
    if pooling_type == 'attention' and os.path.exists(agg_ckpt):
        a = torch.load(agg_ckpt, map_location=device)
        if all(k.startswith('module.') for k in a.keys()):
            a = {k.replace('module.', ''): v for k, v in a.items()}
        aggregator.load_state_dict(a)

    pids = []
    y_true = []
    y_prob = []
    for pid, data in patient_results.items():
        logits = torch.tensor(data['logits']).to(device).unsqueeze(0)
        feats = torch.tensor(np.array(data['features'])).to(device).unsqueeze(0) if need_features else None
        plog = aggregator(logits, feats)
        prob = torch.sigmoid(plog).item()
        pids.append(pid)
        y_true.append(data['label'])
        y_prob.append(prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    youden_idx = np.argmax(j)
    youden_th = float(thresholds[youden_idx])
    y_youden = (y_prob >= youden_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_youden).ravel()
    youden_metrics = {
        'acc': float(accuracy_score(y_true, y_youden)),
        'pre': float(precision_score(y_true, y_youden)),
        'sen': float(recall_score(y_true, y_youden)),
        'f1': float(f1_score(y_true, y_youden)),
        'spe': float(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    }

    f1s = []
    for th in thresholds:
        f1s.append(f1_score(y_true, (y_prob >= th).astype(int)))
    f1s = np.array(f1s)
    best_f1_idx = int(np.nanargmax(f1s))
    f1opt_th = float(thresholds[best_f1_idx])
    y_f1 = (y_prob >= f1opt_th).astype(int)
    f1_metrics = {
        'acc': float(accuracy_score(y_true, y_f1)),
        'pre': float(precision_score(y_true, y_f1)),
        'sen': float(recall_score(y_true, y_f1)),
        'f1': float(f1s[best_f1_idx])
    }

    auc = float(roc_auc_score(y_true, y_prob))

    # save patient probs and summary
    os.makedirs(exp_dir, exist_ok=True)
    import csv
    with open(os.path.join(exp_dir, 'patient_probs.csv'), 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['patient_id', 'label', 'prob'])
        for pid, l, p in zip(pids, y_true, y_prob):
            w.writerow([pid, int(l), float(p)])

    summary = {
        'exp_dir': exp_dir,
        'n_patients': int(len(y_true)),
        'n_positive': int(int(y_true.sum())),
        'auc': auc,
        'youden_threshold': youden_th,
        'youden': youden_metrics,
        'f1opt_threshold': f1opt_th,
        'f1opt': f1_metrics
    }
    with open(os.path.join(exp_dir, 'summary_thresholds.json'), 'w') as jf:
        json.dump(summary, jf, indent=2)

    return summary


def main():
    exps = ['outputs/exp4', 'outputs/exp4-attention', 'outputs/exp4-focal']
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for e in exps:
        if not os.path.isdir(e):
            print(f"Skipping missing {e}")
            continue
        print(f"Evaluating {e}...")
        try:
            results[e] = eval_exp(e, device=device)
        except Exception as ex:
            print(f"Failed evaluating {e}: {ex}")

    # write compare file
    out = 'outputs/exp4-focal/compare_all.txt'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write('Comparison of experiments\n')
        f.write('='*60 + '\n')
        for e, s in results.items():
            f.write(f"{e}: AUC={s['auc']:.4f}, Youden_th={s['youden_threshold']:.3f}, Youden_F1={s['youden']['f1']:.4f}, F1opt_th={s['f1opt_threshold']:.3f}, F1opt={s['f1opt']['f1']:.4f}\n")

    print('Done. Wrote', out)


if __name__ == '__main__':
    main()
