"""
完整评估 + Calibration 脚本

功能：
1. 加载训练好的 EvidenceAggregator
2. 冻结 aggregator，训练 Platt calibration
3. 在 test set 上评估所有指标：
   - AUC, ACC@best, SEN, SPE, PRE, F1
   - ECE, Brier, NLL
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

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from data_utils import PDVolDataset, BagDataset
from models_patch import build_patch_model
from train_evidence_aggregation import (
    EvidenceAggregator, load_patch_encoder, collate_bag_batch
)


class EvidenceCalibrator(nn.Module):
    """Platt Scaling for Evidence Calibration"""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, evidence):
        return torch.sigmoid(self.a * evidence + self.b)


def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find predictions in this bin
        bin_mask = (probs >= bin_lower) & (probs < bin_upper)
        if not bin_mask.any():
            continue

        bin_probs = probs[bin_mask]
        bin_labels = labels[bin_mask]

        # Bin accuracy and confidence
        bin_acc = np.mean(bin_labels)
        bin_conf = np.mean(bin_probs)

        # Add to ECE
        ece += np.abs(bin_acc - bin_conf) * len(bin_labels) / len(probs)

    return ece


def compute_brier_score(probs, labels):
    """Compute Brier Score"""
    return np.mean((probs - labels) ** 2)


def compute_nll(probs, labels):
    """Compute Negative Log Likelihood"""
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return nll


def evaluate_with_calibration(patch_model, evidence_agg, calibrator, loader, device, cfg):
    """评估带calibration的模型"""
    patch_model.eval()
    evidence_agg.eval()
    calibrator.eval()

    all_probs_uncal = []
    all_probs_cal = []
    all_labels = []
    all_evidence = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            patches = batch['patches'].to(device)
            labels = batch['label'].to(device)

            # 提取patch logits
            all_logits = []
            sub_bs = 32
            for i in range(0, len(patches), sub_bs):
                l, _ = patch_model(patches[i:i+sub_bs], return_features=True)
                all_logits.append(l)

            patch_logits = torch.cat(all_logits, dim=0)  # (N,)

            # evidence aggregation
            _, evidence_sum = evidence_agg(patch_logits)

            # uncalibrated probability
            prob_uncal = torch.sigmoid(evidence_sum).item()

            # calibrated probability
            prob_cal = calibrator(evidence_sum).item()

            all_probs_uncal.append(prob_uncal)
            all_probs_cal.append(prob_cal)
            all_labels.append(labels.item())
            all_evidence.append(evidence_sum.item())

    # 转换为numpy
    probs_uncal = np.array(all_probs_uncal)
    probs_cal = np.array(all_probs_cal)
    labels = np.array(all_labels)
    evidence = np.array(all_evidence)

    # 计算指标函数
    def compute_all_metrics(probs, prefix=""):
        auc = roc_auc_score(labels, probs)

        # 找到最佳阈值 (Youden)
        thresholds = np.linspace(0.01, 0.99, 99)
        best_youden = 0.0
        best_thresh = 0.5
        best_metrics = {}

        for thresh in thresholds:
            pred_labels = (probs > thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()

            sen = tp / (tp + fn) if (tp + fn) > 0 else 0
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden = sen + spe - 1

            if youden > best_youden:
                best_youden = youden
                best_thresh = thresh
                acc = accuracy_score(labels, pred_labels)
                pre = precision_score(labels, pred_labels, zero_division=0)
                f1 = f1_score(labels, pred_labels)
                best_metrics = {
                    f'{prefix}acc': acc,
                    f'{prefix}sen': sen,
                    f'{prefix}spe': spe,
                    f'{prefix}pre': pre,
                    f'{prefix}f1': f1,
                    f'{prefix}best_thresh': best_thresh
                }

        return {
            f'{prefix}auc': auc,
            **best_metrics
        }

    # 计算uncalibrated指标
    uncal_metrics = compute_all_metrics(probs_uncal, "uncal_")

    # 计算calibrated指标
    cal_metrics = compute_all_metrics(probs_cal, "cal_")

    # 计算calibration指标
    ece_uncal = compute_ece(probs_uncal, labels)
    ece_cal = compute_ece(probs_cal, labels)
    brier_uncal = compute_brier_score(probs_uncal, labels)
    brier_cal = compute_brier_score(probs_cal, labels)
    nll_uncal = compute_nll(probs_uncal, labels)
    nll_cal = compute_nll(probs_cal, labels)

    results = {
        **uncal_metrics,
        **cal_metrics,
        'ece_uncal': ece_uncal,
        'ece_cal': ece_cal,
        'brier_uncal': brier_uncal,
        'brier_cal': brier_cal,
        'nll_uncal': nll_uncal,
        'nll_cal': nll_cal,
        'evidence_mean': evidence.mean(),
        'evidence_std': evidence.std(),
        'cal_a': calibrator.a.item(),
        'cal_b': calibrator.b.item()
    }

    return results


def train_calibrator(evidence_agg, calibrator, val_loader, device):
    """在validation set上训练calibrator"""
    evidence_agg.eval()
    calibrator.train()

    optimizer = Adam(calibrator.parameters(), lr=1e-2)
    criterion = nn.BCELoss()

    epochs = 50
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0

        for batch in val_loader:
            patches = batch['patches'].to(device)
            labels = batch['label'].to(device)

            # 提取evidence (不需要patch_model，因为aggregator已冻结)
            # 注意：这里假设aggregator的forward返回evidence
            # 但在evaluate中我们用了patch_model来获取patch_logits
            # 这里需要调整：我们需要patch_model来获取evidence

            # 实际上，我们需要patch_model
            # 所以这个函数需要patch_model作为参数

            # 为了简化，我们在主函数中处理
            pass

    # 实际上，训练calibrator需要evidence和labels
    # 我们将在主函数中实现


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--exp1_ckpt', type=str, required=True)
    parser.add_argument('--evidence_ckpt', type=str, required=True)
    parser.add_argument('--test_csv', type=str, default=None, 
                       help='Test manifest CSV path. If None, uses val_csv from config as test set')
    parser.add_argument('--output_dir', type=str, default='outputs/calibration')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 加载配置
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 加载exp1 encoder
    patch_model, _ = load_patch_encoder(args.config, args.exp1_ckpt, device)

    # 加载evidence aggregator
    evidence_agg = EvidenceAggregator().to(device)
    ckpt = torch.load(args.evidence_ckpt, map_location=device)
    evidence_agg.load_state_dict(ckpt)
    evidence_agg.eval()

    # 冻结aggregator
    for param in evidence_agg.parameters():
        param.requires_grad = False

    # 准备validation数据用于训练calibrator
    val_dataset = PDVolDataset(cfg['data']['val_csv'], cfg['data']['input_size'], mode='val')
    patch_size = tuple(cfg['train']['patch_size'])
    stride = tuple(cfg['train']['patch_stride'])
    bag_val = BagDataset(val_dataset, patch_size=patch_size, stride=stride)
    val_loader = DataLoader(bag_val, batch_size=1, shuffle=False, collate_fn=collate_bag_batch, num_workers=0)

    # 准备test数据
    if args.test_csv is None or not os.path.exists(args.test_csv):
        print(f"[INFO] Using validation set as test set (from config)")
        test_csv = cfg['data']['val_csv']
        test_dataset = PDVolDataset(test_csv, cfg['data']['input_size'], mode='test')
        bag_test = BagDataset(test_dataset, patch_size=patch_size, stride=stride)
    else:
        print(f"[INFO] Using separate test set: {args.test_csv}")
        test_dataset = PDVolDataset(args.test_csv, cfg['data']['input_size'], mode='test')
        bag_test = BagDataset(test_dataset, patch_size=patch_size, stride=stride)

    test_loader = DataLoader(bag_test, batch_size=1, shuffle=False, collate_fn=collate_bag_batch, num_workers=0)

    # 创建calibrator
    calibrator = EvidenceCalibrator().to(device)

    # 训练calibrator
    print("[INFO] Training calibrator on validation set...")
    # 收集val evidence和labels
    val_evidence = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Collecting val evidence'):
            patches = batch['patches'].to(device)
            labels = batch['label'].to(device)

            all_logits = []
            sub_bs = 32
            for i in range(0, len(patches), sub_bs):
                l, _ = patch_model(patches[i:i+sub_bs], return_features=True)
                all_logits.append(l)

            patch_logits = torch.cat(all_logits, dim=0)
            _, evidence_sum = evidence_agg(patch_logits)

            val_evidence.append(evidence_sum.item())
            val_labels.append(labels.item())

    val_evidence = torch.tensor(val_evidence, device=device).unsqueeze(1)
    val_labels = torch.tensor(val_labels, device=device).unsqueeze(1)

    # 训练calibrator
    optimizer = Adam(calibrator.parameters(), lr=1e-2)
    criterion = nn.BCELoss()

    epochs = 100
    for epoch in range(epochs):
        calibrator.train()
        optimizer.zero_grad()
        cal_probs = calibrator(val_evidence)
        loss = criterion(cal_probs, val_labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 评估
    print("[INFO] Evaluating on test set...")
    results = evaluate_with_calibration(patch_model, evidence_agg, calibrator, test_loader, device, cfg)

    # 保存结果
    output_file = os.path.join(args.output_dir, 'calibration_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Results saved to {output_file}")
    print("\n=== RESULTS ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == '__main__':
    main()