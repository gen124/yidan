import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_utils import PDVolDataset, PatchDataset
from models_patch import build_patch_model, PatchMILAggregator
from sklearn.metrics import roc_curve
from tqdm import tqdm


def load_cfg(exp_dir, default_cfg='config.yaml'):
    exp_cfg = os.path.join(exp_dir, 'config.yaml')
    path = exp_cfg if os.path.exists(exp_cfg) else default_cfg
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_patient_probs(exp_dir, device, base_model_ckpt=None):
    cfg = load_cfg(exp_dir)
    # load classifier
    model = build_patch_model(cfg).to(device)
    ckpt_path = os.path.join(exp_dir, 'final_best.pth')
    if not os.path.exists(ckpt_path):
        if base_model_ckpt and os.path.exists(base_model_ckpt):
            ckpt_path = base_model_ckpt
        else:
            ckpt_path = os.path.join('outputs', 'exp4', 'final_best.pth')
    ck = torch.load(ckpt_path, map_location=device)
    if isinstance(ck, dict) and 'state_dict' in ck:
        ck = ck['state_dict']
    if all(k.startswith('module.') for k in ck.keys()):
        ck = {k.replace('module.', ''): v for k, v in ck.items()}
    model.load_state_dict(ck)
    model.eval()

    val_dataset = PDVolDataset(manifest_csv=cfg['data']['val_csv'], input_size=cfg['data']['input_size'], mode='val')
    patch_val_dataset = PatchDataset(base_dataset=val_dataset, patch_size=tuple(cfg['train']['patch_size']), stride=tuple(cfg['train']['patch_stride']), mode='pretrain')
    loader = DataLoader(patch_val_dataset, batch_size=64, shuffle=False, num_workers=4)

    pooling_type = cfg['train'].get('pooling', 'topk_mean')
    need_features = pooling_type in ('attention', 'distribution', 'soft_topk', 'quantile')

    patient_results = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Inference {exp_dir}'):
            x = batch['volume'].to(device)
            pids = batch['patient_id']
            labels = batch['label']
            if need_features:
                logits, features = model(x, return_features=True)
                logits = logits.cpu().numpy()
                features = features.cpu().numpy()
            else:
                logits = model(x).cpu().numpy()
                features = [None] * len(logits)
            for i, pid in enumerate(pids):
                if pid not in patient_results:
                    patient_results[pid] = {'logits': [], 'features': [], 'label': int(labels[i])}
                patient_results[pid]['logits'].append(logits[i])
                if need_features:
                    patient_results[pid]['features'].append(features[i])

    # build aggregator
    feat_dim = 0
    if need_features and len(patient_results) > 0:
        first = next(iter(patient_results))
        if len(patient_results[first]['features']) > 0:
            feat_dim = np.array(patient_results[first]['features'])[0].shape[0]

    aggregator = PatchMILAggregator(feat_dim=feat_dim, pooling=pooling_type, topk_percent=cfg['train'].get('topk_percent', 0.15)).to(device)
    # if aggregator weights exist in exp_dir, load
    agg_ckpt = os.path.join(exp_dir, 'aggregator_best.pth')
    if pooling_type == 'attention' and os.path.exists(agg_ckpt):
        a = torch.load(agg_ckpt, map_location=device)
        if all(k.startswith('module.') for k in a.keys()):
            a = {k.replace('module.', ''): v for k, v in a.items()}
        aggregator.load_state_dict(a)

    y_true = []
    pids_list = []
    y_prob = []
    for pid, data in patient_results.items():
        logits = torch.tensor(data['logits']).to(device).unsqueeze(0)
        features = torch.tensor(np.array(data['features'])).to(device).unsqueeze(0) if need_features else None
        patient_logit = aggregator(logits, features)
        prob = torch.sigmoid(patient_logit).item()
        y_true.append(data['label'])
        y_prob.append(prob)
        pids_list.append(pid)

    return pids_list, np.array(y_true), np.array(y_prob)


def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_exp', default='outputs/exp4')
    parser.add_argument('--att_exp', default='outputs/exp4-attention')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    pids_b, y_true_b, y_prob_b = get_patient_probs(args.base_exp, device)
    pids_a, y_true_a, y_prob_a = get_patient_probs(args.att_exp, device)

    # ensure same order
    assert set(pids_b) == set(pids_a)
    pids = pids_b
    # map to same order
    idx_map_a = {pid: i for i, pid in enumerate(pids_a)}
    y_prob_a_aligned = np.array([y_prob_a[idx_map_a[pid]] for pid in pids])

    thr_b = youden_threshold(y_true_b, y_prob_b)
    thr_a = youden_threshold(y_true_b, y_prob_a_aligned)

    pred_b = (y_prob_b >= thr_b).astype(int)
    pred_a = (y_prob_a_aligned >= thr_a).astype(int)

    improved = []
    worsened = []
    for pid, gt, pb, pa in zip(pids, y_true_b, pred_b, pred_a):
        if pb != gt and pa == gt:
            improved.append((pid, int(gt), int(pb), int(pa)))
        if pb == gt and pa != gt:
            worsened.append((pid, int(gt), int(pb), int(pa)))

    out_dir = args.att_exp
    with open(os.path.join(out_dir, 'compare_improved.txt'), 'w') as f:
        for row in improved:
            f.write('\t'.join(map(str, row)) + "\n")
    with open(os.path.join(out_dir, 'compare_worsened.txt'), 'w') as f:
        for row in worsened:
            f.write('\t'.join(map(str, row)) + "\n")

    print('Improved (pid,gt,base_pred,att_pred):', len(improved))
    print('Worsened (pid,gt,base_pred,att_pred):', len(worsened))
    print('Saved to', os.path.join(out_dir, 'compare_improved.txt'), os.path.join(out_dir, 'compare_worsened.txt'))
