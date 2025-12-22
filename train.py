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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
from tqdm import tqdm
import json
from datetime import datetime
import math
import os

from data_utils import PDVolDataset, PatchDataset
from models_patch import ResNet3D_PatchClassifier, PatchMILAggregator, build_patch_model
from metrics import compute_metrics
def _safe_bool(v, default=False):
    try:
        return bool(v)
    except Exception:
        return default

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
    test_bs = min(bs_cfg, max(1, len(dataset)))
    loss_fn = nn.BCEWithLogitsLoss()
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
                with torch.cuda.amp.autocast(enabled=use_amp):
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
        target = int(total * 0.80)  # 目标 80% 总显存
        # 预留当前常驻显存
        reserved = total - free
        budget = max(0, target - reserved)
        max_by_mem = max(1, budget // est_per_sample)
        tuned_bs = max(1, int(min(max_by_mem, bs_cfg * 4)))  # 不超过原始的4倍，避免过大震荡
        print(f"[TUNE] per-sample≈{est_per_sample/1024**2:.1f}MB, total={total/1024**3:.1f}GB, free={free/1024**3:.1f}GB → batch_size={tuned_bs}")
    else:
        print("[TUNE] 跳过显存估计(非CUDA或估计失败), 保持配置 batch_size")

    cpu_cnt = max(1, os.cpu_count() or 8)
    tuned_workers = min( max(4, cpu_cnt // 2), int(cfg['train'].get('num_workers', 8)) )
    # 允许增大到 CPU 一半, 但不超过配置上限
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
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = bool(cfg['train'].get('amp', True)) and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    non_blocking = bool(cfg['train'].get('pin_memory', True)) and (device.type == 'cuda')
    
    for batch in tqdm(loader, desc='Training'):
        x = batch['volume'].to(device, non_blocking=non_blocking)
        y = batch['label'].to(device, non_blocking=non_blocking)
        
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
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
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    use_amp = bool(cfg['train'].get('amp', True)) and (device.type == 'cuda')
    non_blocking = bool(cfg['train'].get('pin_memory', True)) and (device.type == 'cuda')
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            x = batch['volume'].to(device, non_blocking=non_blocking)
            y = batch['label'].to(device, non_blocking=non_blocking)
            with torch.cuda.amp.autocast(enabled=use_amp):
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


def infer_patch_scores(model, dataset, device, batch_size=16):
    """
    对所有patch进行推理,获取预测分数
    
    返回: {(patient_id, patch_idx): score}
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_patch_batch)
    
    patch_scores = {}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Inferring patch scores'):
            x = batch['volume'].to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                logits = model(x)
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


def train_patch_mil(cfg, output_dir='outputs/patch_mil'):
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
    
    # =====================
    # 1. 加载数据集
    # =====================
    print("\n[Stage 1] 加载数据集...")
    train_dataset = PDVolDataset(
        manifest_csv=cfg['data']['train_csv'],
        input_size=cfg['data']['input_size'],
        mode='train',
        transform=cfg['train'].get('augmentations', {}),
        synthetic=False
    )
    val_dataset = PDVolDataset(
        manifest_csv=cfg['data']['val_csv'],
        input_size=cfg['data']['input_size'],
        mode='val',
        transform={},
        synthetic=False
    )
    print(f"  训练集: {len(train_dataset)} 个3D体积")
    print(f"  验证集: {len(val_dataset)} 个3D体积")
    
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
    val_workers = max(0, min(nw, max(1, tuned_workers // 2)))
    train_loader = DataLoader(
        patch_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_patch_batch,
        num_workers=train_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=prefetch if train_workers > 0 else None
    )
    val_loader = DataLoader(
        patch_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patch_batch,
        num_workers=val_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=prefetch if val_workers > 0 else None
    )
    
    # 预训练
    lr_main = float(cfg['train'].get('lr', 1e-4))
    wd_main = float(cfg['train'].get('weight_decay', 1e-5))
    opt = Adam(model.parameters(), lr=lr_main, weight_decay=wd_main)
    use_amp = bool(cfg['train'].get('amp', True)) and (device.type == 'cuda')
    pretrain_epochs = cfg['train'].get('pretrain_epochs', 20)
    best_auc = 0.0
    
    for epoch in range(pretrain_epochs):
        print(f"\n  预训练 Epoch {epoch+1}/{pretrain_epochs}")
        train_loss = train_epoch(model, train_loader, opt, device, cfg)
        metrics, val_loss = evaluate(model, val_loader, device, cfg)
        
        print(f"    训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        print(f"    验证 AUC: {metrics['auc']:.4f}, ACC: {metrics['acc']:.4f}")
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            ckpt_path = os.path.join(output_dir, 'pretrain_best.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"    → 保存最佳模型: {ckpt_path}")
    
    # 加载最佳预训练模型
    best_ckpt = os.path.join(output_dir, 'pretrain_best.pth')
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"\n  加载最佳预训练模型: {best_ckpt}")
    
    # =====================
    # 3. 迭代重标阶段
    # =====================
    n_iters = cfg['train'].get('n_relabel_iters', 3)
    high_conf_threshold = cfg['train'].get('high_conf_threshold', 0.8)
    low_conf_threshold = cfg['train'].get('low_conf_threshold', 0.2)
    
    for iter_idx in range(n_iters):
        print(f"\n[Stage 3.{iter_idx+1}] 迭代重标 - 轮次 {iter_idx+1}/{n_iters}")
        
        # 推理获得patch分数
        patch_scores = infer_patch_scores(model, patch_train_dataset, device, batch_size)
        
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
            prefetch_factor=prefetch if train_workers > 0 else None
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
        torch.save(model.state_dict(), ckpt_path)
        print(f"  保存检查点: {ckpt_path}")
    
    # =====================
    # 4. 分阶段微调阶段
    # =====================
    print(f"\n[Stage 4] 分阶段微调...")
    
    # 阶段1: 冻结骨干,微调分类头
    print("\n  阶段4.1: 冻结骨干,微调分类头...")
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
            torch.save(model.state_dict(), ckpt_path)
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patch级迭代重标训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--out_dir', type=str, default='outputs/patch_mil', help='输出目录')
    parser.add_argument('--synthetic', action='store_true', help='使用合成数据进行测试')
    
    args = parser.parse_args()
    
    cfg = load_cfg(args.config)
    
    # 如果指定了合成数据,则修改配置
    if args.synthetic:
        cfg['data']['train_csv'] = None
        cfg['data']['val_csv'] = None
    
    train_patch_mil(cfg, output_dir=args.out_dir)
