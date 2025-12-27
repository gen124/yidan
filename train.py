"""
Patch级迭代重标训练脚本

完整流程:
1. 预训练阶段: 在3D patch上用患者伪标签预训练
   - 从原始体积切patch -> 每个patch继承患者标签 -> 训练Patch分类器
   
2. 迭代重标阶段: 优化patch标签质量
   - 第1轮: 用预训练模型推理所有patch,得到预测分数
   - 第2轮: 筛选高置信patch (分数>0.8 或 <0.2),重新标注
   - 第3~K轮: 用"高置信重标+低置信伪标签" 继续训练,循环重标
   
3. 分阶段微调: 避免小样本过拟合
   - 阶段1: 冻结骨干,仅微调分类头 (基于重标后的高质量patch)
   - 阶段2: 解冻骨干,联合微调 (使用更小的学习率)
"""

import os
import csv
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
from tqdm import tqdm
import json
from datetime import datetime
import math
import os
import re

from sklearn.metrics import roc_auc_score

from data_utils import PDVolDataset, PatchDataset, BagDataset
from models_patch import ResNet3D_PatchClassifier, PatchMILAggregator, build_patch_model
from metrics import compute_metrics


def _load_model_weights(model, ckpt_path, device):
    """兼容两种格式：纯 state_dict 或 {'state_dict': ...}。"""
    ck = torch.load(ckpt_path, map_location=device)
    if isinstance(ck, dict) and 'state_dict' in ck:
        state_dict = ck['state_dict']
    else:
        state_dict = ck    
    # 处理 DataParallel 包装导致的前缀不匹配
    is_model_dp = isinstance(model, nn.DataParallel)
    is_ckpt_dp = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_model_dp and not is_ckpt_dp:
        # 模型有前缀，权重没有 -> 给权重加前缀
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif not is_model_dp and is_ckpt_dp:
        # 模型没有前缀，权重有 -> 给权重去前缀
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=True)


def _find_latest_relabel_ckpt(output_dir):
    """返回 (ckpt_path, completed_iters)。找不到则返回 (None, 0)。"""
    if not os.path.isdir(output_dir):
        return None, 0
    best_iter = 0
    best_path = None
    for name in os.listdir(output_dir):
        m = re.match(r"relabel_iter(\d+)\.pth$", name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > best_iter:
            best_iter = idx
            best_path = os.path.join(output_dir, name)
    return best_path, best_iter
def _safe_bool(v, default=False):
    try:
        return bool(v)
    except Exception:
        return default


class FocalLoss(nn.Module):
    """Binary focal loss for logits.

    Args:
        alpha (float): weight for positive class. If None, set 0.5.
        gamma (float): focusing parameter.
        reduction (str): 'mean'|'sum'|'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha if alpha is not None else 0.5
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: Tensor of shape (N,1) or (N,) ; targets: same shape
        probs = torch.sigmoid(logits)
        targets = targets.type_as(probs)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_fn(cfg):
    """Return a loss function according to cfg['train']['loss']

    Supported: 'bce' (default), 'focal'
    """
    loss_name = str(cfg['train'].get('loss', 'bce')).lower()
    if loss_name == 'focal':
        alpha = float(cfg['train'].get('focal_alpha', 0.25))
        gamma = float(cfg['train'].get('focal_gamma', 2.0))
        return FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
    else:
        return nn.BCEWithLogitsLoss()

def auto_tune_batch_and_workers(model, dataset, device, cfg):
    """
    基于一次试跑估计每样本显存占用, 动态放大 batch_size 直到接近目标显存占用,
    同时根据 CPU 核心数给出合适的 num_workers。
    返回: tuned_batch_size, tuned_num_workers
    """
    bs_cfg = int(cfg['train'].get('batch_size', 2))
    use_amp = _safe_bool(cfg['train'].get('amp', True)) and (device.type == 'cuda')

    # 估计每样本显存占用
    est_per_sample = None
    test_bs = min(bs_cfg, 8) # 只需要很小的 batch 就能估计显存
    loss_fn = get_loss_fn(cfg)
    non_blocking = _safe_bool(cfg['train'].get('pin_memory', True)) and (device.type == 'cuda')

    if device.type == 'cuda':
        try:
            tmp_loader = DataLoader(dataset, batch_size=test_bs, shuffle=False, collate_fn=collate_patch_batch, num_workers=0)
            batch = next(iter(tmp_loader))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model.eval()
            with torch.no_grad():
                x = batch['volume'].to(device, non_blocking=non_blocking)
                y = batch['label'].to(device, non_blocking=non_blocking)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(x)
                    loss = loss_fn(logits, y.view_as(logits))
                _ = float(loss.item())
            peak = torch.cuda.max_memory_allocated(device)
            est_per_sample = max(1, peak // max(1, test_bs))
        except Exception:
            est_per_sample = None

    tuned_bs = bs_cfg
    if device.type == 'cuda' and est_per_sample is not None and est_per_sample > 0:
        free, total = torch.cuda.mem_get_info(device)
        
        # 如果是 DataParallel，显存总量翻倍（假设双卡相同）
        n_gpus = 1
        if isinstance(model, nn.DataParallel):
            n_gpus = len(model.device_ids)
        
        target = int(total * 0.70 * n_gpus)  # 降低到 70% 显存，预留更多缓冲
        # 预留当前常驻显存
        reserved = (total - free) * n_gpus
        budget = max(0, target - reserved)
        max_by_mem = max(1, budget // est_per_sample)
        
        # 限制在配置的 4 倍以内，且最高不超过 1024，平衡显存与稳定性
        tuned_bs = max(1, int(min(max_by_mem, bs_cfg * 4, 1024)))
        # 确保是 n_gpus 的倍数，方便平分
        tuned_bs = (tuned_bs // n_gpus) * n_gpus
        
        print(f"[TUNE] per-sample≈{est_per_sample/1024**2:.1f}MB, total={total*n_gpus/1024**3:.1f}GB, free={free*n_gpus/1024**3:.1f}GB → batch_size={tuned_bs}")
    else:
        print("[TUNE] 跳过显存估计(非CUDA或估计失败), 保持配置 batch_size")

    cpu_cnt = max(1, os.cpu_count() or 8)
    tuned_workers = min( 4, int(cfg['train'].get('num_workers', 4)) )
    # 限制在 4 个 worker，避免内存占用过高
    print(f"[TUNE] CPU cores={cpu_cnt} → num_workers={tuned_workers}")
    print(f"[TUNE] CPU cores={cpu_cnt} → num_workers={tuned_workers}")
    return int(tuned_bs), int(tuned_workers)


def load_cfg(path='config.yaml'):
    """加载配置文件"""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def collate_patch_batch(batch):
    """Patch数据集的collate函数"""
    vols = torch.stack([s['volume'] for s in batch], dim=0)
    labels = torch.tensor([s['label'] for s in batch], dtype=torch.float32)
    patient_ids = [s['patient_id'] for s in batch]
    patch_idxs = [s['patch_idx'] for s in batch]
    ids = [s['id'] for s in batch]
    
    return {
        'volume': vols,
        'label': labels,
        'patient_id': patient_ids,
        'patch_idx': patch_idxs,
        'id': ids
    }


def train_epoch(model, loader, opt, device, cfg):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    loss_fn = get_loss_fn(cfg)
    use_amp = bool(cfg['train'].get('amp', True)) and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    non_blocking = bool(cfg['train'].get('pin_memory', True)) and (device.type == 'cuda')
    
    for batch in tqdm(loader, desc='Training'):
        x = batch['volume'].to(device, non_blocking=non_blocking)
        y = batch['label'].to(device, non_blocking=non_blocking)
        
        opt.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y.view_as(logits))
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / max(1, n_batches)
    return avg_loss


def evaluate(model, loader, device, cfg):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    loss_fn = get_loss_fn(cfg)
    total_loss = 0.0
    n_batches = 0
    use_amp = bool(cfg['train'].get('amp', True)) and (device.type == 'cuda')
    non_blocking = bool(cfg['train'].get('pin_memory', True)) and (device.type == 'cuda')
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            x = batch['volume'].to(device, non_blocking=non_blocking)
            y = batch['label'].to(device, non_blocking=non_blocking)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y.view_as(logits))
            total_loss += loss.item()
            n_batches += 1
            
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()
            
            all_preds.extend(probs)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # compute_metrics expects (y_true, y_prob)
    metrics = compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / max(1, n_batches)
    
    return metrics, avg_loss


def infer_patch_scores(model, dataset, device, batch_size=16, num_workers=0, cfg=None):
    """
    对所有patch进行推理,获取预测分数
    
    返回: {(patient_id, patch_idx): score}
    """
    model.eval()
    # 强制使用 num_workers=0 以确保内存绝对安全，推理阶段 GPU 是瓶颈，CPU 串行提取 patch 足够
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_patch_batch,
        num_workers=0,
        pin_memory=(device.type=='cuda')
    )
    
    patch_scores = {}

    # decide whether to use feature-derived score for relabeling (default: use logits)
    use_feature_score = False
    feature_score_type = 'l2'
    mlp_ckpt = None
    mlp_hidden = None
    if cfg is not None:
        use_feature_score = bool(cfg['train'].get('relabel_use_feature_score', False))
        feature_score_type = cfg['train'].get('feature_score_type', 'l2')
        mlp_ckpt = cfg['train'].get('feature_score_mlp_ckpt', None)
        mlp_hidden = cfg['train'].get('feature_score_mlp_hidden', None)

    # if using mlp as score and ckpt provided, build small mlp
    mlp = None
    if use_feature_score and feature_score_type == 'mlp':
        if mlp_ckpt is None:
            print('[WARN] feature_score_type=mlp but feature_score_mlp_ckpt not provided; falling back to l2')
            use_feature_score = True
            feature_score_type = 'l2'
        else:
            # build small MLP: feat_dim -> hidden -> 1
            # We'll infer feat_dim on the fly from first batch
            mlp = None

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inferring patch scores'):
            x = batch['volume'].to(device)
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                if use_feature_score:
                    logits, features = model(x, return_features=True)
                else:
                    logits = model(x)

            if use_feature_score:
                # features: (B, D)
                feats = features.cpu()
                if feature_score_type == 'l2':
                    scores = feats.norm(p=2, dim=1).numpy()
                else:
                    # mlp unsupported fallback: use l2
                    scores = feats.norm(p=2, dim=1).numpy()
                for i, (pid, pidx) in enumerate(zip(batch['patient_id'], batch['patch_idx'])):
                    patch_scores[(pid, pidx)] = float(scores[i])
            else:
                probs = torch.sigmoid(logits).cpu().numpy()
                for i, (pid, pidx) in enumerate(zip(batch['patient_id'], batch['patch_idx'])):
                    patch_scores[(pid, pidx)] = float(probs[i])
    
    return patch_scores


def relabel_patches(patch_scores, high_conf_threshold=0.8, low_conf_threshold=0.2):
    """
    迭代重标: 筛选高置信patch并重新标注
    
    逻辑:
    - 分数 > high_conf_threshold: 标注为 1 (高置信正样本)
    - 分数 < low_conf_threshold: 标注为 0 (高置信负样本)
    - 分数在中间: 保留原伪标签
    
    返回:
    - refined_labels: {(patient_id, patch_idx): new_label}
    - stats: 重标统计信息
    """
    refined_labels = {}
    high_conf_count = 0
    low_conf_count = 0
    
    for (pid, pidx), score in patch_scores.items():
        if score > high_conf_threshold:
            refined_labels[(pid, pidx)] = 1
            high_conf_count += 1
        elif score < low_conf_threshold:
            refined_labels[(pid, pidx)] = 0
            low_conf_count += 1
        # 否则不在refined_labels中,将使用原伪标签
    
    stats = {
        'high_conf_pos': high_conf_count,
        'high_conf_neg': low_conf_count,
        'total_relabeled': high_conf_count + low_conf_count
    }
    
    return refined_labels, stats


def train_patch_mil(cfg, output_dir='outputs/patch_mil', resume=None):
    """
    完整的Patch级迭代重标训练流程
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(cfg['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    # 加速选项
    if device.type == 'cuda' and cfg['train'].get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
    seed = cfg['train'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 输出目录: {output_dir}")
    if resume:
        print(f"[INFO] 续训模式: {resume}")
    
    # =====================
    # 1. 加载数据集
    # =====================
    print("\n[Stage 1] 加载数据集...")
    # 如果配置中的 train_csv 为 None，认为是 synthetic 调试模式
    use_synthetic = (cfg['data'].get('train_csv') is None)
    train_dataset = PDVolDataset(
        manifest_csv=cfg['data']['train_csv'],
        input_size=cfg['data']['input_size'],
        mode='train',
        transform=cfg['train'].get('augmentations', {}),
        synthetic=use_synthetic
    )
    val_dataset = PDVolDataset(
        manifest_csv=cfg['data']['val_csv'],
        input_size=cfg['data']['input_size'],
        mode='val',
        transform={},
        synthetic=use_synthetic
    )
    print(f"  训练集: {len(train_dataset)} 个3D体积")
    print(f"  验证集: {len(val_dataset)} 个3D体积")

    # [NEW] 预缓存数据到内存，避免多进程时每个 worker 都占用一份内存
    print("[INFO] 正在预缓存数据到内存 (约占用 15GB)...")
    from tqdm import tqdm
    for i in tqdm(range(len(train_dataset)), desc="Caching Train"):
        _ = train_dataset[i]
    for i in tqdm(range(len(val_dataset)), desc="Caching Val"):
        _ = val_dataset[i]
    
    # =====================
    # 2. 预训练阶段
    # =====================
    print("\n[Stage 2] Patch级预训练...")
    patch_size = tuple(cfg['train'].get('patch_size', [48, 48, 32]))
    stride = tuple(cfg['train'].get('patch_stride', patch_size))
    
    # 构建Patch数据集 (使用患者伪标签)
    patch_train_dataset = PatchDataset(
        base_dataset=train_dataset,
        patch_size=patch_size,
        stride=stride,
        patch_labels=None,  # 预训练阶段无重标
        mode='pretrain'
    )
    patch_val_dataset = PatchDataset(
        base_dataset=val_dataset,
        patch_size=patch_size,
        stride=stride,
        patch_labels=None,
        mode='pretrain'
    )
    print(f"  生成的Patch数: 训练={len(patch_train_dataset)}, 验证={len(patch_val_dataset)}")
    
    # 构建模型
    model = build_patch_model(cfg).to(device)

    # 多GPU支持
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"[INFO] 检测到 {torch.cuda.device_count()} 个 GPU，启用 DataParallel")
        model = nn.DataParallel(model)
    
    # 创建数据加载器(自动调参)
    batch_size_cfg = int(cfg['train'].get('batch_size', 2))
    # 先基于预训练patch数据集做一次显存/CPU估计
    try:
        tuned_bs, tuned_workers = auto_tune_batch_and_workers(model, patch_train_dataset, device, cfg)
    except Exception as e:
        print(f"[TUNE] 自动调参失败: {e}. 使用配置值继续")
        tuned_bs, tuned_workers = batch_size_cfg, int(cfg['train'].get('num_workers', 4))
    batch_size = tuned_bs
    nw = int(cfg['train'].get('num_workers', 4))
    pin_mem = bool(cfg['train'].get('pin_memory', True)) and (device.type == 'cuda')
    prefetch = int(cfg['train'].get('prefetch_factor', 2))
    persistent = bool(cfg['train'].get('persistent_workers', False)) and nw > 0
    # 训练时优先稳态，不超过配置的并行度
    train_workers = max(0, min(nw, tuned_workers))
    val_workers = 0  # 验证阶段强制 0 以节省内存
    train_loader = DataLoader(
        patch_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_patch_batch,
        num_workers=train_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=1 if train_workers > 0 else None
    )
    val_loader = DataLoader(
        patch_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patch_batch,
        num_workers=val_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=1 if val_workers > 0 else None
    )
    
    # ===== 续训：尝试加载已有 checkpoint =====
    resume_ckpt_path = None
    completed_relabel_iters = 0

    if resume is not None:
        if str(resume).lower() == 'auto':
            # 优先从最新的 relabel_iterK 恢复，其次从预训练最优恢复
            relabel_ckpt, completed = _find_latest_relabel_ckpt(output_dir)
            if relabel_ckpt is not None:
                resume_ckpt_path = relabel_ckpt
                completed_relabel_iters = int(completed)
            else:
                pre_ckpt = os.path.join(output_dir, 'pretrain_best.pth')
                if os.path.exists(pre_ckpt):
                    resume_ckpt_path = pre_ckpt
                    completed_relabel_iters = 0
        else:
            # 指定文件路径
            if os.path.exists(resume):
                resume_ckpt_path = resume
            else:
                raise FileNotFoundError(f"resume checkpoint not found: {resume}")

    if resume_ckpt_path is not None:
        _load_model_weights(model, resume_ckpt_path, device)
        print(f"[RESUME] 已加载权重: {resume_ckpt_path}")
    else:
        print("[RESUME] 未启用/未发现可用checkpoint，将从头开始预训练")

    # 预训练
    lr_main = float(cfg['train'].get('lr', 1e-4))
    wd_main = float(cfg['train'].get('weight_decay', 1e-5))
    opt = Adam(model.parameters(), lr=lr_main, weight_decay=wd_main)
    use_amp = bool(cfg['train'].get('amp', True)) and (device.type == 'cuda')
    pretrain_epochs = cfg['train'].get('pretrain_epochs', 20)
    best_auc = 0.0

    if resume_ckpt_path is None:
        # 从头预训练
        for epoch in range(pretrain_epochs):
            print(f"\n  预训练 Epoch {epoch+1}/{pretrain_epochs}")
            train_loss = train_epoch(model, train_loader, opt, device, cfg)
            metrics, val_loss = evaluate(model, val_loader, device, cfg)

            print(f"    训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            print(f"    验证 AUC: {metrics['auc']:.4f}, ACC: {metrics['acc']:.4f}")

            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                ckpt_path = os.path.join(output_dir, 'pretrain_best.pth')
            # 处理 DataParallel 包装
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckpt_path)
        best_ckpt = os.path.join(output_dir, 'pretrain_best.pth')
        if os.path.exists(best_ckpt):
            _load_model_weights(model, best_ckpt, device)
            print(f"\n  加载最佳预训练模型: {best_ckpt}")
    else:
        # 已从 relabel_iterK.pth 加载，跳过预训练阶段
        print("[RESUME] 检测到已完成预训练，跳过 Stage 2")
    
    # =====================
    # 3. 迭代重标阶段
    # =====================
    n_iters = cfg['train'].get('n_relabel_iters', 3)
    high_conf_threshold = cfg['train'].get('high_conf_threshold', 0.8)
    low_conf_threshold = cfg['train'].get('low_conf_threshold', 0.2)
    
    patch_train_dataset_refined = None
    train_loader_refined = None

    start_iter = int(completed_relabel_iters)
    if start_iter > 0:
        print(f"[RESUME] 已完成迭代重标轮次: {start_iter}/{n_iters}，将从第 {start_iter+1} 轮继续")

    for iter_idx in range(start_iter, n_iters):
        print(f"\n[Stage 3.{iter_idx+1}] 迭代重标 - 轮次 {iter_idx+1}/{n_iters}")
        
        # 推理获得patch分数
        patch_scores = infer_patch_scores(model, patch_train_dataset, device, batch_size, num_workers=train_workers, cfg=cfg)
        
        # 重标高置信patch
        refined_labels, stats = relabel_patches(
            patch_scores,
            high_conf_threshold=high_conf_threshold,
            low_conf_threshold=low_conf_threshold
        )
        
        print(f"  重标统计: 高置信正样本={stats['high_conf_pos']}, "
              f"高置信负样本={stats['high_conf_neg']}, "
              f"总重标={stats['total_relabeled']}")
        
        # 用重标后的标签创建新数据集
        patch_train_dataset_refined = PatchDataset(
            base_dataset=train_dataset,
            patch_size=patch_size,
            stride=stride,
            patch_labels=refined_labels,
            mode='finetune'
        )
        
        train_loader_refined = DataLoader(
            patch_train_dataset_refined,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_patch_batch,
            num_workers=train_workers,
            pin_memory=pin_mem,
            persistent_workers=persistent,
            prefetch_factor=1 if train_workers > 0 else None
        )
        
        # 微调
        finetune_epochs = cfg['train'].get('finetune_epochs', 5)
        lr_ft = float(cfg['train'].get('lr', 1e-4)) * 0.5
        wd_ft = float(cfg['train'].get('weight_decay', 1e-5))
        opt_finetune = Adam(
            model.parameters(),
            lr=lr_ft,
            weight_decay=wd_ft
        )
        
        for epoch in range(finetune_epochs):
            train_loss = train_epoch(model, train_loader_refined, opt_finetune, device, cfg)
            metrics, val_loss = evaluate(model, val_loader, device, cfg)
            
            if (epoch + 1) % 2 == 0:
                print(f"    微调 Epoch {epoch+1}/{finetune_epochs} - "
                      f"训练损失: {train_loss:.4f}, 验证AUC: {metrics['auc']:.4f}")
        
        # 保存迭代检查点
        ckpt_path = os.path.join(output_dir, f'relabel_iter{iter_idx+1}.pth')
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, ckpt_path)
        print(f"  保存检查点: {ckpt_path}")

    # 如果 Stage 3 被完全跳过(例如 n_iters=0 或 resume 到 n_iters), 仍需构建 refined 数据集供 Stage 4 使用
    if patch_train_dataset_refined is None:
        print("\n[Stage 3] 未执行迭代重标循环，重算一次 refined labels 以进入 Stage 4...")
        patch_scores = infer_patch_scores(model, patch_train_dataset, device, batch_size, num_workers=train_workers, cfg=cfg)
        refined_labels, stats = relabel_patches(
            patch_scores,
            high_conf_threshold=high_conf_threshold,
            low_conf_threshold=low_conf_threshold
        )
        print(f"  重标统计(补算): 高置信正样本={stats['high_conf_pos']}, 高置信负样本={stats['high_conf_neg']}, 总重标={stats['total_relabeled']}")
        patch_train_dataset_refined = PatchDataset(
            base_dataset=train_dataset,
            patch_size=patch_size,
            stride=stride,
            patch_labels=refined_labels,
            mode='finetune'
        )
        train_loader_refined = DataLoader(
            patch_train_dataset_refined,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_patch_batch,
            num_workers=train_workers,
            pin_memory=pin_mem,
            persistent_workers=persistent,
            prefetch_factor=1 if train_workers > 0 else None
        )
    
    # =====================
    # 4. 分阶段微调阶段
    # =====================
    print(f"\n[Stage 4] 分阶段微调...")
    
    # 阶段1: 冻结骨干,微调分类头
    print("\n  阶段4.1: 冻结骨干,微调分类头...")
    if isinstance(model, nn.DataParallel):
        model.module.freeze_backbone()
    else:
        model.freeze_backbone()
    
    lr_head = float(cfg['train'].get('lr', 1e-4))
    wd_head = float(cfg['train'].get('weight_decay', 1e-5))
    opt_head = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_head,
        weight_decay=wd_head
    )
    
    head_finetune_epochs = cfg['train'].get('head_finetune_epochs', 5)
    for epoch in range(head_finetune_epochs):
        train_loss = train_epoch(model, train_loader_refined, opt_head, device, cfg)
        metrics, val_loss = evaluate(model, val_loader, device, cfg)
        print(f"    Epoch {epoch+1}/{head_finetune_epochs} - 训练损失: {train_loss:.4f}, 验证AUC: {metrics['auc']:.4f}")
    
    # 阶段2: 解冻骨干,联合微调
    print("\n  阶段4.2: 解冻骨干,联合微调...")
    if isinstance(model, nn.DataParallel):
        model.module.unfreeze_backbone()
    else:
        model.unfreeze_backbone()
    
    lr_full = float(cfg['train'].get('lr', 1e-4)) * 0.1
    wd_full = float(cfg['train'].get('weight_decay', 1e-5))
    opt_full = Adam(
        model.parameters(),
        lr=lr_full,  # 更小的学习率
        weight_decay=wd_full
    )
    
    final_epochs = cfg['train'].get('final_epochs', 10)
    best_final_auc = 0.0
    
    for epoch in range(final_epochs):
        train_loss = train_epoch(model, train_loader_refined, opt_full, device, cfg)
        metrics, val_loss = evaluate(model, val_loader, device, cfg)
        
        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}/{final_epochs} - 训练损失: {train_loss:.4f}, 验证AUC: {metrics['auc']:.4f}")
        
        if metrics['auc'] > best_final_auc:
            best_final_auc = metrics['auc']
            ckpt_path = os.path.join(output_dir, 'final_best.pth')
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckpt_path)
    
    print(f"\n[INFO] 训练完成! 最佳模型: {output_dir}/final_best.pth")
    
    # 保存配置和元数据
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'output_dir': output_dir,
        'patch_size': patch_size,
        'patch_stride': stride,
        'n_patches_train': len(patch_train_dataset),
        'n_patches_val': len(patch_val_dataset),
        'pretrain_epochs': pretrain_epochs,
        'n_relabel_iters': n_iters,
        'final_auc': float(best_final_auc)
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] 元数据已保存: {metadata_path}")


def collate_bag_batch(batch):
    """Bag数据集的collate函数 (B=1)"""
    return {
        'patches': batch[0]['patches'], # (N, C, D, H, W)
        'label': torch.tensor([batch[0]['label']], dtype=torch.float32),
        'patient_id': batch[0]['patient_id']
    }

def train_attention_stage(cfg, output_dir, base_model_path):
    """
    Stage 5: 冻结 ResNet, 训练 Attention Aggregator (Bag-level)
    """
    print("\n[Stage 5] 训练 Attention Aggregator (Bag-level)...")
    device = torch.device(cfg['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # 1. 加载基础模型 (ResNet)
    resnet = build_patch_model(cfg).to(device)
    _load_model_weights(resnet, base_model_path, device)
    resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = False
    
    # 2. 初始化 Aggregator
    # 获取特征维度 (试跑一个 patch)
    with torch.no_grad():
        dummy_x = torch.zeros(1, cfg['model']['in_channels'], *cfg['train']['patch_size'][::-1]).to(device)
        _, feat = resnet(dummy_x, return_features=True)
        feat_dim = feat.shape[1]
    
    # use pooling type from config so we can train aggregator matching experiment setting
    pooling_type = cfg['train'].get('pooling', 'attention')
    aggregator = PatchMILAggregator(feat_dim=feat_dim, pooling=pooling_type).to(device)
    
    # 3. 准备数据
    train_dataset = PDVolDataset(cfg['data']['train_csv'], cfg['data']['input_size'], mode='train')
    val_dataset = PDVolDataset(cfg['data']['val_csv'], cfg['data']['input_size'], mode='val')
    
    bag_train = BagDataset(train_dataset, patch_size=tuple(cfg['train']['patch_size']), stride=tuple(cfg['train']['patch_stride']))
    bag_val = BagDataset(val_dataset, patch_size=tuple(cfg['train']['patch_size']), stride=tuple(cfg['train']['patch_stride']))
    
    # Bag-level 训练通常 batch_size=1 (因为每个病人的 patch 数量不等)
    train_loader = DataLoader(bag_train, batch_size=1, shuffle=True, collate_fn=collate_bag_batch, num_workers=0)
    val_loader = DataLoader(bag_val, batch_size=1, shuffle=False, collate_fn=collate_bag_batch, num_workers=0)
    
    opt = Adam(aggregator.parameters(), lr=float(cfg['train'].get('lr', 1e-4)))
    loss_fn = get_loss_fn(cfg)
    
    best_auc = 0.0
    epochs = cfg['train'].get('final_epochs', 10)
    
    for epoch in range(epochs):
        aggregator.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Stage 5 - Epoch {epoch+1}"):
            patches = batch['patches'].to(device) # (N, C, D, H, W)
            label = batch['label'].to(device)
            
            with torch.no_grad():
                # 提取所有 patch 的特征
                # 如果 N 太大，分批提取
                all_logits = []
                all_feats = []
                sub_bs = 32
                for i in range(0, len(patches), sub_bs):
                    l, f = resnet(patches[i:i+sub_bs], return_features=True)
                    all_logits.append(l)
                    all_feats.append(f)
                logits = torch.cat(all_logits, dim=0).unsqueeze(0) # (1, N)
                feats = torch.cat(all_feats, dim=0).unsqueeze(0)   # (1, N, D)
            
            opt.zero_grad()
            patient_logit = aggregator(logits, feats)
            loss = loss_fn(patient_logit, label)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        # 验证
        aggregator.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for batch in val_loader:
                patches = batch['patches'].to(device)
                all_logits, all_feats = [], []
                for i in range(0, len(patches), 32):
                    l, f = resnet(patches[i:i+32], return_features=True)
                    all_logits.append(l)
                    all_feats.append(f)
                logits = torch.cat(all_logits, dim=0).unsqueeze(0)
                feats = torch.cat(all_feats, dim=0).unsqueeze(0)
                
                p_logit = aggregator(logits, feats)
                y_prob.append(torch.sigmoid(p_logit).item())
                y_true.append(batch['label'].item())
        
        auc = roc_auc_score(y_true, y_prob)
        print(f"  Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f}, Val AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            save_name = f"aggregator_{pooling_type}_best.pth"
            torch.save(aggregator.state_dict(), os.path.join(output_dir, save_name))
            
    print(f"[INFO] Stage 5 完成! 最佳 Aggregator AUC: {best_auc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patch级迭代重标训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--out_dir', type=str, default='outputs/patch_mil', help='输出目录')
    parser.add_argument('--synthetic', action='store_true', help='使用合成数据进行测试')
    parser.add_argument('--resume', type=str, default=None, help="续训: auto 或 checkpoint 路径")
    parser.add_argument('--stage5', action='store_true', help='仅运行 Stage 5 (Attention 微调)')
    parser.add_argument('--base_model', type=str, default=None, help='Stage 5 所需的基础模型路径')
    
    args = parser.parse_args()
    
    cfg = load_cfg(args.config)
    
    # 如果指定了合成数据,则修改配置
    if args.synthetic:
        cfg['data']['train_csv'] = None
        cfg['data']['val_csv'] = None
    
    if args.stage5:
        if not args.base_model:
            # 尝试自动寻找
            args.base_model = os.path.join(args.out_dir, 'final_best.pth')
        train_attention_stage(cfg, args.out_dir, args.base_model)
    else:
        train_patch_mil(cfg, output_dir=args.out_dir, resume=args.resume)

