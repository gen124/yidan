"""
通用校准脚本 - 支持任意Aggregator类型

可以对 attention, distribution, max 等任意pooling方法进行校准
"""

import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, List

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from data_utils import PDVolDataset, BagDataset
from models_patch import ResNet3D_PatchClassifier, PatchMILAggregator, build_patch_model
from train_evidence_aggregation import EvidenceAggregator, collate_bag_batch


class UniversalCalibrator(nn.Module):
    """通用Platt Scaling校准器"""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, logit):
        return torch.sigmoid(self.a * logit + self.b)


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        bin_mask = (probs >= bin_lower) & (probs < bin_upper)
        if not bin_mask.any():
            continue

        bin_acc = np.mean(labels[bin_mask])
        bin_conf = np.mean(probs[bin_mask])

        ece += np.abs(bin_acc - bin_conf) * len(bin_mask) / len(probs)

    return ece


def compute_brier_score(probs, labels):
    """Brier Score"""
    return np.mean((probs - labels) ** 2)


def compute_nll(probs, labels):
    """Negative Log Likelihood"""
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))


def load_aggregator_model(cfg_path, ckpt_path, aggregator_ckpt_path, pooling_type, device):
    """加载patch model和aggregator"""
    # 加载配置
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 构建patch model
    patch_model = build_patch_model(cfg).to(device)

    # 构建aggregator
    feat_dim = cfg['model']['base_channels'] * 8  # ResNet输出维度
    if pooling_type == 'evidence':
        aggregator = EvidenceAggregator().to(device)
    else:
        aggregator = PatchMILAggregator(
            feat_dim=feat_dim,
            pooling=pooling_type,
            topk_percent=cfg['train'].get('topk_percent', 0.15)
        ).to(device)

    # 加载patch model权重
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    model_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            model_dict[k[7:]] = v
        else:
            model_dict[k] = v

    patch_model.load_state_dict({
        k: v for k, v in model_dict.items()
        if not k.startswith('aggregator.')
    }, strict=False)

    # 加载aggregator权重
    if aggregator_ckpt_path and os.path.exists(aggregator_ckpt_path):
        agg_state = torch.load(aggregator_ckpt_path, map_location=device)
        aggregator.load_state_dict(agg_state)
        print(f"[INFO] 加载aggregator权重: {aggregator_ckpt_path}")
    else:
        print(f"[WARNING] aggregator权重文件不存在: {aggregator_ckpt_path}")

    # 冻结模型
    for param in patch_model.parameters():
        param.requires_grad = False
    for param in aggregator.parameters():
        param.requires_grad = False

    print(f"[INFO] {pooling_type} 模型已加载并冻结")

    return patch_model, aggregator, cfg


def evaluate_with_calibration(patch_model, aggregator, calibrator, loader, device, cfg, pooling_type):
    """评估带calibration的模型"""
    patch_model.eval()
    aggregator.eval()
    calibrator.eval()

    all_logits = []
    all_probs_uncal = []
    all_probs_cal = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Evaluating {pooling_type}'):
            patches = batch['patches'].to(device)
            labels = batch['label'].to(device)

            # 提取patch features和logits
            all_patch_logits = []
            all_features = []
            sub_bs = 32
            for i in range(0, len(patches), sub_bs):
                l, f = patch_model(patches[i:i+sub_bs], return_features=True)
                all_patch_logits.append(l)
                all_features.append(f)

            patch_logits = torch.cat(all_patch_logits, dim=0)  # (N,)
            patch_features = torch.cat(all_features, dim=0)  # (N, D)

            # 根据pooling类型调用aggregator
            if pooling_type in ['max', 'topk_mean', 'mean', 'evidence']:
                # logit-based pooling
                patient_logit = aggregator(patch_logits.unsqueeze(0))
                if isinstance(patient_logit, tuple):
                    patient_logit = patient_logit[0]
            else:
                # feature-based pooling
                patient_logit = aggregator(patch_logits.unsqueeze(0), patch_features.unsqueeze(0))

            # 转换为概率
            patient_prob_uncal = torch.sigmoid(patient_logit).item()
            patient_prob_cal = calibrator(patient_logit).item()

            all_logits.append(patient_logit.item())
            all_probs_uncal.append(patient_prob_uncal)
            all_probs_cal.append(patient_prob_cal)
            all_labels.append(labels.item())

    # 转换为numpy
    logits = np.array(all_logits)
    probs_uncal = np.array(all_probs_uncal)
    probs_cal = np.array(all_probs_cal)
    labels = np.array(all_labels)

    # 计算AUC
    auc = roc_auc_score(labels, probs_uncal)

    # 计算ECE, Brier, NLL
    ece_uncal = compute_ece(probs_uncal, labels)
    ece_cal = compute_ece(probs_cal, labels)
    brier_uncal = compute_brier_score(probs_uncal, labels)
    brier_cal = compute_brier_score(probs_cal, labels)
    nll_uncal = compute_nll(probs_uncal, labels)
    nll_cal = compute_nll(probs_cal, labels)

    def _best_metrics(probs):
        thresholds = np.linspace(0.01, 0.99, 99)
        best = {
            'f1': 0,
            'thresh': 0.5,
            'acc': 0,
            'sen': 0,
            'spe': 0,
            'pre': 0
        }
        for thresh in thresholds:
            pred_labels = (probs > thresh).astype(int)

            tp = ((pred_labels == 1) & (labels == 1)).sum()
            tn = ((pred_labels == 0) & (labels == 0)).sum()
            fp = ((pred_labels == 1) & (labels == 0)).sum()
            fn = ((pred_labels == 0) & (labels == 1)).sum()

            if tp + fn == 0 or tp + fp == 0:
                continue

            sen = tp / (tp + fn)
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0
            pre = tp / (tp + fp)
            acc = (tp + tn) / len(labels)
            f1 = 2 * pre * sen / (pre + sen)

            if f1 > best['f1']:
                best.update({'f1': f1, 'thresh': thresh, 'acc': acc, 'sen': sen, 'spe': spe, 'pre': pre})
        return best

    best_uncal = _best_metrics(probs_uncal)
    best_cal = _best_metrics(probs_cal)

    results = {
        'pooling_type': pooling_type,
        'auc': auc,
        'ece_uncal': ece_uncal,
        'ece_cal': ece_cal,
        'brier_uncal': brier_uncal,
        'brier_cal': brier_cal,
        'nll_uncal': nll_uncal,
        'nll_cal': nll_cal,
        'acc': best_cal['acc'],
        'sen': best_cal['sen'],
        'spe': best_cal['spe'],
        'pre': best_cal['pre'],
        'f1': best_cal['f1'],
        'best_thresh': best_cal['thresh'],
        'acc_uncal': best_uncal['acc'],
        'sen_uncal': best_uncal['sen'],
        'spe_uncal': best_uncal['spe'],
        'pre_uncal': best_uncal['pre'],
        'f1_uncal': best_uncal['f1'],
        'best_thresh_uncal': best_uncal['thresh'],
        'logit_mean': logits.mean(),
        'logit_std': logits.std(),
        'cal_a': calibrator.a.item(),
        'cal_b': calibrator.b.item(),
        'num_samples': len(labels),
        'y_true': labels.tolist(),
        'y_prob_uncal': probs_uncal.tolist(),
        'y_prob_cal': probs_cal.tolist()
    }

    return results


def train_calibrator(patch_model, aggregator, calibrator, val_loader, device, cfg, pooling_type):
    """在validation set上训练calibrator"""
    patch_model.eval()
    aggregator.eval()
    calibrator.train()

    # 收集validation logits
    val_logits = []
    val_labels = []

    print("[INFO] 收集validation logits用于训练calibrator...")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Collecting val logits'):
            patches = batch['patches'].to(device)
            labels = batch['label'].to(device)

            all_patch_logits = []
            all_features = []
            sub_bs = 32
            for i in range(0, len(patches), sub_bs):
                l, f = patch_model(patches[i:i+sub_bs], return_features=True)
                all_patch_logits.append(l)
                all_features.append(f)

            patch_logits = torch.cat(all_patch_logits, dim=0)
            patch_features = torch.cat(all_features, dim=0)

            # 调用aggregator
            if pooling_type in ['max', 'topk_mean', 'mean', 'evidence']:
                patient_logit = aggregator(patch_logits.unsqueeze(0))
                if isinstance(patient_logit, tuple):
                    patient_logit = patient_logit[0]
            else:
                patient_logit = aggregator(patch_logits.unsqueeze(0), patch_features.unsqueeze(0))

            val_logits.append(patient_logit.item())
            val_labels.append(labels.item())

    val_logits = torch.tensor(val_logits, device=device).unsqueeze(1)
    val_labels = torch.tensor(val_labels, device=device).unsqueeze(1)

    # 训练calibrator
    optimizer = Adam(calibrator.parameters(), lr=1e-2)
    criterion = nn.BCELoss()

    epochs = 100
    for epoch in range(epochs):
        calibrator.train()
        optimizer.zero_grad()
        cal_probs = calibrator(val_logits)
        loss = criterion(cal_probs, val_labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return calibrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--ckpt_path', type=str, required=True, help='patch model checkpoint路径')
    parser.add_argument('--aggregator_ckpt', type=str, default=None, help='aggregator checkpoint路径')
    parser.add_argument('--pooling_type', type=str, required=True,
                       choices=['distribution', 'max', 'topk_mean', 'mean', 'attention', 'soft_topk', 'quantile', 'evidence'],
                       help='Pooling类型')
    parser.add_argument('--test_csv', type=str, default=None, help='测试集CSV路径')
    parser.add_argument('--output_dir', type=str, default='outputs/calibration')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 加载模型
    patch_model, aggregator, cfg = load_aggregator_model(
        args.config, args.ckpt_path, args.aggregator_ckpt, args.pooling_type, device
    )

    # 准备validation数据用于训练calibrator
    val_dataset = PDVolDataset(cfg['data']['val_csv'], cfg['data']['input_size'], mode='val')
    patch_size = tuple(cfg['train']['patch_size'])
    stride = tuple(cfg['train']['patch_stride'])
    bag_val = BagDataset(val_dataset, patch_size=patch_size, stride=stride)
    val_loader = DataLoader(bag_val, batch_size=1, shuffle=False, collate_fn=collate_bag_batch, num_workers=0)

    # 准备test数据
    if args.test_csv and os.path.exists(args.test_csv):
        test_dataset = PDVolDataset(args.test_csv, cfg['data']['input_size'], mode='test')
    else:
        test_dataset = PDVolDataset(cfg['data']['val_csv'], cfg['data']['input_size'], mode='val')

    bag_test = BagDataset(test_dataset, patch_size=patch_size, stride=stride)
    test_loader = DataLoader(bag_test, batch_size=1, shuffle=False, collate_fn=collate_bag_batch, num_workers=0)

    # 创建calibrator
    calibrator = UniversalCalibrator().to(device)

    # 训练calibrator
    print("[INFO] 训练calibrator...")
    calibrator = train_calibrator(patch_model, aggregator, calibrator, val_loader, device, cfg, args.pooling_type)

    # 评估
    print("[INFO] 评估校准效果...")
    results = evaluate_with_calibration(patch_model, aggregator, calibrator, test_loader, device, cfg, args.pooling_type)

    # 保存结果
    output_file = os.path.join(args.output_dir, f'{args.pooling_type}_calibration_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] 结果已保存到 {output_file}")
    print("\n=== 校准结果 ===")
    print(f"Pooling类型: {results['pooling_type']}")
    print(f"AUC:         {results['auc']:.4f}")
    print(f"ECE:         {results['ece_uncal']:.4f} → {results['ece_cal']:.4f}")
    print(f"Brier:       {results['brier_uncal']:.4f} → {results['brier_cal']:.4f}")
    print(f"NLL:         {results['nll_uncal']:.4f} → {results['nll_cal']:.4f}")
    print(f"F1:          {results['f1_uncal']:.4f} → {results['f1']:.4f} (阈值: {results['best_thresh_uncal']:.2f} → {results['best_thresh']:.2f})")
    print(f"ACC:         {results['acc_uncal']:.4f} → {results['acc']:.4f}")
    print(f"SEN:         {results['sen_uncal']:.4f} → {results['sen']:.4f}")
    print(f"SPE:         {results['spe_uncal']:.4f} → {results['spe']:.4f}")
    print(f"PRE:         {results['pre_uncal']:.4f} → {results['pre']:.4f}")
    print(f"校准参数:    a={results['cal_a']:.3f}, b={results['cal_b']:.3f}")


if __name__ == '__main__':
    main()