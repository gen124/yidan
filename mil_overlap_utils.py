"""
Patch重叠下的MIL聚合工具

支持有重叠Patch场景下的多层聚合:
  Layer 1: 体素级加权投票
  Layer 2: ROI级聚合
  Layer 3: 患者级决策
"""

import numpy as np
from typing import Dict, Tuple, List
import torch

from models_patch import PatchMILAggregator


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def weighted_vote_aggregation(
    patch_logits: Dict[Tuple[int, int, int], float],
    patch_coords: Dict[Tuple[int, int, int], Tuple[int, int, int, int, int, int]],
    volume_shape: Tuple[int, int, int],
    edge_weight_method: str = 'linear'
) -> np.ndarray:
    """
    体素级加权投票：处理重叠Patch
    
    Args:
        patch_logits: {(h,w,d): logit_score}
        patch_coords: {(h,w,d): (h_start, w_start, d_start, ph, pw, pd)}
        volume_shape: (D, H, W)
        edge_weight_method: 'linear' | 'cosine' | 'uniform'
    
    Returns:
        voxel_probs: (D, H, W) 体素级概率热图
    
    原理：
      每个体素如果被多个Patch覆盖，使用加权投票
      中心体素权重高（更确定），边缘体素权重低
    """
    D, H, W = volume_shape
    
    # 创建体素级的投票和权重
    voxel_votes = np.zeros((D, H, W), dtype=np.float32)
    voxel_weights = np.zeros((D, H, W), dtype=np.float32)
    
    for (h, w, d), logit in patch_logits.items():
        h_start, w_start, d_start, ph, pw, pd = patch_coords[(h, w, d)]
        prob = sigmoid(logit)
        
        # 遍历Patch内的所有体素
        for di in range(pd):
            for hi in range(ph):
                for wi in range(pw):
                    voxel_d = d_start + di
                    voxel_h = h_start + hi
                    voxel_w = w_start + wi
                    
                    # 边界检查
                    if not (0 <= voxel_d < D and 0 <= voxel_h < H and 0 <= voxel_w < W):
                        continue
                    
                    # 计算体素在Patch中的位置权重
                    # 中心体素权重高，边缘体素权重低
                    if edge_weight_method == 'linear':
                        # 边缘距离
                        edge_dist = min(
                            di, pd - 1 - di,
                            hi, ph - 1 - hi,
                            wi, pw - 1 - wi
                        )
                        edge_weight = (edge_dist + 1) / (max(ph, pw, pd) // 2 + 1)
                    
                    elif edge_weight_method == 'cosine':
                        # 余弦衰减
                        rel_d = di / (pd - 1) if pd > 1 else 0.5
                        rel_h = hi / (ph - 1) if ph > 1 else 0.5
                        rel_w = wi / (pw - 1) if pw > 1 else 0.5
                        
                        # 距离中心越近，余弦值越接近1
                        edge_weight = (
                            (np.cos((rel_d - 0.5) * np.pi) + 1) / 2 *
                            (np.cos((rel_h - 0.5) * np.pi) + 1) / 2 *
                            (np.cos((rel_w - 0.5) * np.pi) + 1) / 2
                        )
                    
                    else:  # uniform
                        edge_weight = 1.0
                    
                    # 累积投票
                    voxel_votes[voxel_d, voxel_h, voxel_w] += prob * edge_weight
                    voxel_weights[voxel_d, voxel_h, voxel_w] += edge_weight
    
    # 计算体素级平均概率
    voxel_probs = np.divide(
        voxel_votes,
        voxel_weights,
        where=voxel_weights > 0,
        out=np.zeros_like(voxel_votes)
    )
    
    return voxel_probs


def roi_based_aggregation(
    voxel_probs: np.ndarray,
    aal_mask: np.ndarray,
    aggregation_type: str = 'mean'
) -> Dict[int, float]:
    """
    ROI级聚合：将体素级概率按脑区聚合
    
    Args:
        voxel_probs: (D, H, W) 体素级概率
        aal_mask: (D, H, W) AAL脑图谱标签
        aggregation_type: 'mean' | 'topk' | 'max' | 'median'
    
    Returns:
        roi_scores: {roi_id: score}
    """
    roi_scores = {}
    
    for roi_id in np.unique(aal_mask):
        if roi_id <= 0:  # 跳过背景
            continue
        
        roi_mask = (aal_mask == roi_id)
        roi_probs = voxel_probs[roi_mask]
        
        if len(roi_probs) == 0:
            continue
        
        if aggregation_type == 'mean':
            roi_score = roi_probs.mean()
        
        elif aggregation_type == 'topk':
            # TopK Mean: 最关键的20%体素的平均
            k = max(1, int(len(roi_probs) * 0.2))
            roi_score = np.sort(roi_probs)[-k:].mean()
        
        elif aggregation_type == 'max':
            # 检测局部强异常
            roi_score = roi_probs.max()
        
        elif aggregation_type == 'median':
            # 鲁棒的中位数
            roi_score = np.median(roi_probs)
        
        else:
            roi_score = roi_probs.mean()
        
        roi_scores[roi_id] = float(roi_score)
    
    return roi_scores


def patient_level_decision(
    roi_scores: Dict[int, float],
    aggregation: str = 'topk_mean',
    topk_ratio: float = 0.15
) -> float:
    """
    患者级决策：从ROI分数到患者诊断概率
    
    Args:
        roi_scores: {roi_id: score}
        aggregation: 'topk_mean' | 'max' | 'mean' | 'median'
        topk_ratio: TopK的比例
    
    Returns:
        patient_prob: (0, 1) 患者诊断概率
    """
    if len(roi_scores) == 0:
        return 0.5
    
    scores = np.array(list(roi_scores.values()))
    
    if aggregation == 'topk_mean':
        k = max(1, int(len(scores) * topk_ratio))
        patient_prob = np.sort(scores)[-k:].mean()
    elif aggregation == 'max':
        patient_prob = scores.max()
    elif aggregation == 'median':
        patient_prob = np.median(scores)
    else:  # mean
        patient_prob = scores.mean()
    
    return float(np.clip(patient_prob, 0, 1))


def analyze_patch_importance(
    patch_logits: Dict[Tuple[int, int, int], float],
    aal_mask: np.ndarray,
    top_k: int = 5
) -> List[Tuple[Tuple[int, int, int], float, int]]:
    """
    分析最重要的Patch
    
    Args:
        patch_logits: {(h,w,d): logit}
        aal_mask: (D, H, W)
        top_k: 返回Top K个Patch
    
    Returns:
        [(patch_coords, score, dominant_roi_id), ...]
    """
    patch_importance = []
    
    for patch_coords, logit in patch_logits.items():
        prob = sigmoid(logit)
        
        # 找出Patch中的主要ROI
        h, w, d = patch_coords
        ph, pw, pd = 48, 48, 32  # 假设标准尺寸
        
        patch_aal = aal_mask[d:min(d+pd, aal_mask.shape[0]),
                              h:min(h+ph, aal_mask.shape[1]),
                              w:min(w+pw, aal_mask.shape[2])]
        
        if patch_aal.size == 0:
            dominant_roi = 0
        else:
            roi_ids = patch_aal[patch_aal > 0]
            if len(roi_ids) == 0:
                dominant_roi = 0
            else:
                dominant_roi = np.argmax(np.bincount(roi_ids))
        
        patch_importance.append((patch_coords, prob, dominant_roi))
    
    # 按概率降序排序
    patch_importance.sort(key=lambda x: x[1], reverse=True)
    
    return patch_importance[:top_k]


def consistency_check(
    voxel_votes: np.ndarray,
    voxel_weights: np.ndarray,
    threshold: float = 0.8
) -> np.ndarray:
    """
    一致性检查：多个Patch对同一体素的看法是否一致
    
    Args:
        voxel_votes: 体素的概率投票和
        voxel_weights: 体素的权重和
        threshold: 一致性阈值
    
    Returns:
        consistency_score: (D, H, W) 一致性得分
    """
    # 标准差小 = 多个Patch意见一致
    voxel_probs = np.divide(
        voxel_votes,
        voxel_weights,
        where=voxel_weights > 0,
        out=np.zeros_like(voxel_votes)
    )
    
    # 近似计算方差：如果多个Patch给出相同分数，方差小
    # 简单策略：如果一个体素被K个Patch覆盖，权重接近K*avg_weight
    consistency = np.clip(voxel_weights / voxel_weights.max(), 0, 1)
    
    return consistency


def _infer_patch_logits_and_coords(
    model,
    patient_volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    patch_stride: Tuple[int, int, int],
    device: str,
    verbose: bool = False,
    return_features: bool = False
):
    """推理所有patch,返回logit与坐标"""
    _, D, H, W = patient_volume.shape
    ph, pw, pd = patch_size
    sh, sw, sd = patch_stride

    patch_logits: Dict[Tuple[int, int, int], float] = {}
    patch_coords: Dict[Tuple[int, int, int], Tuple[int, int, int, int, int, int]] = {}
    patch_count = 0
    patch_features: Dict[Tuple[int, int, int], torch.Tensor] = {} if return_features else None

    model.eval()
    with torch.no_grad():
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                for d in range(0, D - pd + 1, sd):
                    patch = patient_volume[:, d:d+pd, h:h+ph, w:w+pw]
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                    if return_features:
                        logit, feat = model(patch_tensor, return_features=True)
                        patch_features[(h, w, d)] = feat.squeeze(0)
                        logit = logit.squeeze()
                    else:
                        logit = model(patch_tensor).squeeze()
                    patch_logits[(h, w, d)] = logit.item()
                    patch_coords[(h, w, d)] = (h, w, d, ph, pw, pd)
                    patch_count += 1

    if verbose:
        print(f"  完成: {patch_count} 个Patch推理")

    return patch_logits, patch_coords, patch_count, patch_features


def prepare_overlap_report(
    patch_logits: Dict[Tuple[int, int, int], float],
    voxel_probs: np.ndarray,
    roi_scores: Dict[int, float],
    patient_prob: float,
    aal_mask: np.ndarray
) -> Dict:
    """
    生成Patch重叠的分析报告

    Returns:
        {
            'num_patches': int,
            'voxel_coverage': float,
            'patch_stats': {...},
            'top_rois': [...],
            'patient_diagnosis': float,
            'summary': str
        }
    """
    num_patches = len(patch_logits)
    voxel_coverage = float((voxel_probs > 0).sum() / voxel_probs.size) if voxel_probs.size else 0.0

    if num_patches > 0:
        logits = np.array(list(patch_logits.values()), dtype=np.float32)
        probs = sigmoid(logits)
        patch_stats = {
            'num_patches': num_patches,
            'mean_logit': float(logits.mean()),
            'max_logit': float(logits.max()),
            'min_logit': float(logits.min()),
            'mean_prob': float(probs.mean()),
            'high_prob_ratio': float((probs > 0.7).sum() / len(probs))
        }
    else:
        patch_stats = {
            'num_patches': 0,
            'mean_logit': 0.0,
            'max_logit': 0.0,
            'min_logit': 0.0,
            'mean_prob': 0.0,
            'high_prob_ratio': 0.0
        }

    top_rois = sorted(roi_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        'num_patches': num_patches,
        'voxel_coverage': voxel_coverage,
        'patch_stats': patch_stats,
        'top_rois': top_rois,
        'patient_diagnosis': float(patient_prob),
        'summary': (
            f"患者诊断概率: {patient_prob:.3f} | "
            f"Patch数: {num_patches} | "
            f"体素覆盖率: {voxel_coverage:.1%} | "
            f"高风险ROI: {len([x for x in roi_scores.values() if x > 0.7])}"
        )
    }


# ============================================================================
# 完整管道函数
# ============================================================================

def complete_overlap_mil_pipeline(
    model,
    patient_volume: np.ndarray,
    aal_mask: np.ndarray,
    patch_size: Tuple[int, int, int] = (48, 48, 32),
    patch_stride: Tuple[int, int, int] = (24, 24, 16),
    device: str = 'cpu',
    verbose: bool = True,
    pooling: str = 'topk_mean',
    topk_percent: float = 0.15
) -> Dict:
    """
    自适应的Patch MIL聚合管道 (重叠/非重叠自动切换)
    """

    if verbose:
        print("[MIL Pipeline] 开始处理Patch样本...")
        print(f"  Patch尺寸: {patch_size}")
        print(f"  Patch步长: {patch_stride}")

    overlap = any(st < sz for st, sz in zip(patch_stride, patch_size))
    if verbose:
        mode_msg = "重叠" if overlap else "无重叠"
        print(f"  检测到: {mode_msg}配置")
        if overlap:
            overlap_percent = (1 - patch_stride[0] / patch_size[0]) * 100
            print(f"  Overlap比例约: {overlap_percent:.0f}%")

    C, D, H, W = patient_volume.shape

    if verbose:
        print("[MIL Pipeline] Layer 1: 推理所有Patch...")

    patch_logits, patch_coords, patch_count, patch_features = _infer_patch_logits_and_coords(
        model,
        patient_volume,
        patch_size,
        patch_stride,
        device,
        verbose=False,
        return_features=(not overlap and pooling == 'attention')
    )

    if verbose:
        print(f"  完成: {patch_count} 个Patch推理")

    if overlap:
        if verbose:
            print("[MIL Pipeline] Layer 2: 体素级加权投票...")

        voxel_probs = weighted_vote_aggregation(
            patch_logits,
            patch_coords,
            (D, H, W),
            edge_weight_method='linear'
        )

        if verbose:
            coverage = (voxel_probs > 0).sum() / voxel_probs.size
            print(f"  完成: 体素级热图 {voxel_probs.shape}")
            print(f"  体素覆盖率: {coverage:.1%}")

        if verbose:
            print("[MIL Pipeline] Layer 3: ROI级聚合...")
        roi_scores = roi_based_aggregation(voxel_probs, aal_mask, aggregation_type='mean')

        if verbose:
            print(f"  完成: {len(roi_scores)} 个脑区聚合")

        if verbose:
            print("[MIL Pipeline] Layer 4: 患者级决策...")
        patient_prob = patient_level_decision(roi_scores, aggregation='topk_mean')

        if verbose:
            print(f"  患者诊断概率: {patient_prob:.3f}")

    else:
        if verbose:
            print("[MIL Pipeline] 使用非重叠聚合 (TopK Mean)...")

        voxel_probs = np.zeros((D, H, W), dtype=np.float32)
        for (h, w, d), logit in patch_logits.items():
            h_start, w_start, d_start, ph, pw, pd = patch_coords[(h, w, d)]
            prob = float(sigmoid(logit))
            voxel_probs[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] = prob

        feat_dim = model.fc.in_features if pooling == 'attention' and hasattr(model, 'fc') else 128
        aggregator = PatchMILAggregator(feat_dim=feat_dim, pooling=pooling, topk_percent=topk_percent).to(device)
        aggregator.eval()

        if len(patch_logits) == 0:
            patient_prob = 0.5
        else:
            logits_tensor = torch.tensor(list(patch_logits.values()), dtype=torch.float32, device=device).unsqueeze(0)
            features_tensor = None
            if pooling == 'attention':
                if patch_features is None:
                    raise ValueError("attention pooling需要patch_features")
                features_list = [patch_features[k] for k in patch_logits.keys()]
                features_tensor = torch.stack(features_list, dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                patient_logit = aggregator(logits_tensor, features_tensor).squeeze().item()
            patient_prob = float(sigmoid(patient_logit))

        roi_scores = roi_based_aggregation(voxel_probs, aal_mask, aggregation_type='mean')

        if verbose:
            print(f"  ROI数量: {len(roi_scores)} | 患者诊断概率: {patient_prob:.3f}")

    top_patches = analyze_patch_importance(patch_logits, aal_mask, top_k=5)

    report = prepare_overlap_report(
        patch_logits,
        voxel_probs,
        roi_scores,
        patient_prob,
        aal_mask
    )
    report['aggregation_mode'] = 'overlap' if overlap else 'non_overlap'

    if verbose:
        print(report['summary'])

    return {
        'patient_prob': patient_prob,
        'voxel_heatmap': voxel_probs,
        'roi_scores': roi_scores,
        'patch_logits': patch_logits,
        'top_patches': top_patches,
        'report': report,
        'overlap_mode': 'overlap' if overlap else 'non_overlap'
    }
