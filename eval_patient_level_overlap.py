import os
import yaml
import numpy as np
import torch
from data_utils import PDVolDataset
from models_patch import build_patch_model
from mil_overlap_utils import complete_overlap_mil_pipeline
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def eval_patient_level_overlap(exp_dir='outputs/exp4', config_path='config.yaml'):
    exp_config = os.path.join(exp_dir, 'config.yaml')
    if os.path.exists(exp_config):
        config_path = exp_config

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    # load model
    model = build_patch_model(cfg).to(device)
    ckpt_path = os.path.join(exp_dir, 'final_best.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device)
    if isinstance(ck, dict) and 'state_dict' in ck:
        state_dict = ck['state_dict']
    else:
        state_dict = ck
    # strip module. if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load val dataset (PDVolDataset gives full volumes)
    val_dataset = PDVolDataset(manifest_csv=cfg['data']['val_csv'], input_size=cfg['data']['input_size'], mode='val')
    print(f"[INFO] 验证集患者数: {len(val_dataset)}")

    y_true = []
    y_prob = []

    patch_size = tuple(cfg['train'].get('patch_size'))
    patch_stride = tuple(cfg['train'].get('patch_stride'))
    pooling = cfg['train'].get('pooling', 'topk_mean')
    topk_percent = cfg['train'].get('topk_percent', 0.15)

    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        vol_t = sample['volume']  # torch tensor (C, D, H, W)
        aal = sample['aal']  # numpy or None
        label = int(sample['label'])
        pid = sample['id']

        patient_vol = vol_t.numpy() if isinstance(vol_t, torch.Tensor) else vol_t

        try:
            aal_mask = (aal.astype(np.int32) if aal is not None else np.zeros(patient_vol.shape[1:], dtype=np.int32))
            res = complete_overlap_mil_pipeline(
                model=model,
                patient_volume=patient_vol,
                aal_mask=aal_mask,
                patch_size=patch_size,
                patch_stride=patch_stride,
                device=device,
                verbose=False,
                pooling=pooling,
                topk_percent=topk_percent
            )
            prob = float(res['patient_prob'])
        except Exception as e:
            print(f"[WARN] {pid} pipeline failed: {e}; fallback to 0.5")
            prob = 0.5

        y_true.append(label)
        y_prob.append(prob)
        print(f"{pid}: label={label} prob={prob:.4f}")

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]
    y_pred = (y_prob >= best_thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob)

    print("\n" + "="*30)
    print(f"患者级评估结果 (overlap pipeline)\n实验: {exp_dir}")
    print(f"聚合方式: overlap pipeline")
    print(f"患者总数: {len(y_true)}")
    print(f"正样本数: {y_true.sum()}")
    print(f"负样本数: {len(y_true)-y_true.sum()}")
    print("-"*30)
    print(f"AUC: {auc:.4f}")
    print(f"ACC: {acc:.4f}")
    print(f"SEN: {sen:.4f}")
    print(f"PRE: {pre:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"SPE: {spe:.4f}")
    print("="*30)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='outputs/exp4')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    eval_patient_level_overlap(args.exp_dir, args.config)
