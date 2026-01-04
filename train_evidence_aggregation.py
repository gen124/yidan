"""
训练Evidence Accumulation优化exp1

方案3：Soft-count / Evidence Accumulation
- 冻结exp1的encoder
- 替换aggregator为EvidenceAggregator
- 累积patch-level证据进行bag-level决策
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

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from data_utils import PDVolDataset, BagDataset
from models_patch import ResNet3D_PatchClassifier, build_patch_model
from metrics import compute_metrics


def collate_bag_batch(batch):
    """Bag数据集的collate函数 (B=1)"""
    return {
        'patches': batch[0]['patches'],  # (N, C, D, H, W)
        'label': torch.tensor([batch[0]['label']], dtype=torch.float32),
        'patient_id': batch[0]['patient_id']
    }


class EvidenceAggregator(nn.Module):
    """Evidence Accumulation Aggregator"""

    def __init__(self, tau=0.3, alpha=10.0, learnable_params=True):
        super().__init__()
        if learnable_params:
            self.tau = nn.Parameter(torch.tensor(tau))
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer('tau', torch.tensor(tau))
            self.register_buffer('alpha', torch.tensor(alpha))

        self.linear = nn.Linear(1, 1)

    def forward(self, patch_logits, normalize=True):
        """
        Args:
            patch_logits: (N,) patch-level logits
            normalize: whether to normalize by patch count
        Returns:
            patient_logit: scalar
            evidence_sum: accumulated evidence
        """
        patch_probs = torch.sigmoid(patch_logits)

        # soft evidence function
        evidence = torch.sigmoid(self.alpha * (patch_probs - self.tau))

        # accumulate evidence
        E = evidence.sum(dim=-1, keepdim=True)

        # optional normalization by patch count
        if normalize:
            N = torch.tensor(patch_logits.shape[-1], dtype=E.dtype, device=E.device)
            E = E / N

        # patient-level logit
        patient_logit = self.linear(E)

        return patient_logit, E.squeeze()


def load_patch_encoder(cfg_path, ckpt_path, device):
    """加载任意patch model的encoder部分"""
    # 加载配置
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 构建模型
    patch_model = build_patch_model(cfg).to(device)

    # 加载checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # 处理DataParallel前缀
    model_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            model_dict[k[7:]] = v  # 移除'module.'前缀
        else:
            model_dict[k] = v

    # 只加载patch model参数
    patch_model.load_state_dict({
        k: v for k, v in model_dict.items()
        if not k.startswith('aggregator.')
    }, strict=False)

    # 冻结所有参数
    for param in patch_model.parameters():
        param.requires_grad = False

    print("[INFO] Patch encoder已加载并冻结")

    return patch_model, cfg


def evaluate_evidence_aggregation(patch_model, evidence_agg, loader, device, cfg):
    """评估evidence aggregation性能"""
    patch_model.eval()
    evidence_agg.eval()

    all_probs = []
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
            patient_logit, evidence_sum = evidence_agg(patch_logits)
            patient_prob = torch.sigmoid(patient_logit)

            all_probs.append(patient_prob.item())
            all_labels.append(labels.item())
            all_evidence.append(evidence_sum.item())

    # 计算指标
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    evidence = np.array(all_evidence)

    # 原始指标
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, (probs > 0.5).astype(int))
    f1 = f1_score(labels, (probs > 0.5).astype(int))

    # sweep threshold找到最佳F1
    thresholds = np.linspace(0.1, 0.9, 17)
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in thresholds:
        pred_labels = (probs > thresh).astype(int)
        current_f1 = f1_score(labels, pred_labels)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

    results = {
        'auc': auc,
        'acc': acc,
        'f1': f1,
        'f1_best': best_f1,
        'best_thresh': best_thresh,
        'evidence_mean': evidence.mean(),
        'evidence_std': evidence.std(),
        'tau': evidence_agg.tau.item(),
        'alpha': evidence_agg.alpha.item()
    }

    return results


def train_evidence_aggregation(cfg_path, patch_ckpt, output_dir, device='cuda'):
    """训练evidence aggregation"""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    # 加载patch encoder
    patch_model, cfg = load_patch_encoder(cfg_path, patch_ckpt, device)

    # 创建evidence aggregator
    evidence_agg = EvidenceAggregator(tau=0.3, alpha=10.0, learnable_params=True).to(device)

    # 准备数据
    train_dataset = PDVolDataset(cfg['data']['train_csv'], cfg['data']['input_size'], mode='train')
    val_dataset = PDVolDataset(cfg['data']['val_csv'], cfg['data']['input_size'], mode='val')

    patch_size = tuple(cfg['train']['patch_size'])
    stride = tuple(cfg['train']['patch_stride'])

    bag_train = BagDataset(train_dataset, patch_size=patch_size, stride=stride)
    bag_val = BagDataset(val_dataset, patch_size=patch_size, stride=stride)

    train_loader = DataLoader(bag_train, batch_size=1, shuffle=True, collate_fn=collate_bag_batch, num_workers=0)
    val_loader = DataLoader(bag_val, batch_size=1, shuffle=False, collate_fn=collate_bag_batch, num_workers=0)

    # 计算正样本权重
    train_labels = [bag_train[i]['label'] for i in range(len(bag_train))]
    pos_weight = len(train_labels) / sum(train_labels) if sum(train_labels) > 0 else 1.0
    pos_weight = torch.tensor(pos_weight, device=device)

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(evidence_agg.parameters(), lr=1e-3, weight_decay=1e-4)

    # 训练
    best_f1 = 0.0
    epochs = 15  # evidence aggregation收敛快
    patience = 5
    no_improve_count = 0

    for epoch in range(epochs):
        evidence_agg.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            patches = batch['patches'].to(device)
            label = batch['label'].to(device)

            # 前向传播
            all_logits = []
            sub_bs = 32
            for i in range(0, len(patches), sub_bs):
                l, _ = patch_model(patches[i:i+sub_bs], return_features=True)
                all_logits.append(l)

            patch_logits = torch.cat(all_logits, dim=0)  # (N,)

            patient_logit, _ = evidence_agg(patch_logits)

            # 计算损失
            loss = criterion(patient_logit, label)
            train_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        val_results = evaluate_evidence_aggregation(patch_model, evidence_agg, val_loader, device, cfg)

        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}")
        print(f"  AUC={val_results['auc']:.4f}, ACC={val_results['acc']:.4f}, F1={val_results['f1']:.4f} (best F1={val_results['f1_best']:.4f} @ thresh={val_results['best_thresh']:.2f})")
        print(f"  Evidence: mean={val_results['evidence_mean']:.2f}, std={val_results['evidence_std']:.2f}")
        print(f"  Params: tau={val_results['tau']:.3f}, alpha={val_results['alpha']:.1f}")

        # 保存最佳模型
        current_f1 = val_results['f1_best']
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(evidence_agg.state_dict(), os.path.join(output_dir, 'evidence_agg_best.pth'))
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 保存最终结果
    final_results = evaluate_evidence_aggregation(patch_model, evidence_agg, val_loader, device, cfg)
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\n[INFO] 训练完成!")
    print(f"最佳F1: {best_f1:.4f}")
    print("结果已保存到:", output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练Evidence Accumulation优化exp1')
    parser.add_argument('--exp1_config', type=str, default='outputs/new_refine/exp1/config.yaml',
                       help='exp1的配置文件路径')
    parser.add_argument('--exp1_ckpt', type=str, default='outputs/new_refine/exp1/final_best.pth',
                       help='exp1的checkpoint路径')
    parser.add_argument('--output_dir', type=str, default='outputs/exp1_evidence_agg',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')

    args = parser.parse_args()

    train_evidence_aggregation(
        cfg_path=args.exp1_config,
        exp1_ckpt=args.exp1_ckpt,
        output_dir=args.output_dir,
        device=args.device
    )