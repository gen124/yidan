"""
Patch重叠场景下的使用示例

演示如何在inference.py中集成分层MIL聚合
"""

from mil_overlap_utils import (
    complete_overlap_mil_pipeline,
    analyze_patch_importance,
    prepare_overlap_report,
    sigmoid
)
import numpy as np
import torch
import json
import os


def run_inference_with_overlap_support(
    model_ckpt: str,
    qsm_path: str,
    t1_path: str,
    aal_path: str,
    cfg: dict,
    out_dir: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    升级版推理函数：支持Patch重叠的分层MIL聚合
    
    这是对原有inference.py的补充，专门处理重叠Patch场景
    
    Args:
        model_ckpt: 模型检查点路径
        qsm_path: QSM脑图像路径
        t1_path: T1脑图像路径
        aal_path: AAL脑图谱标签路径
        cfg: 配置字典
        out_dir: 输出目录
        device: 'cuda' | 'cpu'
    
    返回:
        results: 包含诊断概率、热图、ROI分数等
    """
    
    print("="*70)
    print("推理管道 (支持Patch重叠)")
    print("="*70)
    
    # ========== Step 1: 加载数据 ==========
    print("\n[Step 1] 加载数据...")
    
    # 加载脑图像（这里假设使用nibabel或scipy）
    try:
        import nibabel as nib
        qsm = nib.load(qsm_path).get_fdata().astype(np.float32)
        t1 = nib.load(t1_path).get_fdata().astype(np.float32)
        aal_mask = nib.load(aal_path).get_fdata().astype(np.int32)
    except ImportError:
        # 如果没有nibabel，尝试使用scipy
        from scipy import io
        qsm = io.loadmat(qsm_path)['qsm'].astype(np.float32)
        t1 = io.loadmat(t1_path)['t1'].astype(np.float32)
        aal_mask = io.loadmat(aal_path)['aal'].astype(np.int32)
    
    # 读取配置中的输入尺寸
    input_shape = cfg['data'].get('input_size', [192, 192, 128])
    
    print(f"  QSM shape: {qsm.shape}")
    print(f"  T1 shape: {t1.shape}")
    print(f"  AAL shape: {aal_mask.shape}")
    print(f"  输入尺寸: {input_shape}")
    
    # ========== Step 2: 数据预处理 ==========
    print("\n[Step 2] 数据预处理...")
    
    # 标准化
    def zscore_normalize(x):
        mean, std = x.mean(), x.std()
        if std > 0:
            x = (x - mean) / std
        return x
    
    qsm = zscore_normalize(qsm)
    t1 = zscore_normalize(t1)
    
    # 中心裁剪或填充到目标尺寸
    def center_crop_or_pad(img, target_shape):
        """中心裁剪或填充"""
        current_shape = img.shape
        output = np.zeros(target_shape, dtype=img.dtype)
        
        # 计算起始位置
        start = tuple((c - t) // 2 if c > t else 0 
                      for c, t in zip(current_shape, target_shape))
        end = tuple(start[i] + target_shape[i] 
                    for i in range(len(target_shape)))
        
        # 裁剪
        slices_src = tuple(slice(max(0, start[i]), min(current_shape[i], end[i])) 
                           for i in range(len(target_shape)))
        slices_dst = tuple(slice(0, end[i] - start[i]) 
                           for i in range(len(target_shape)))
        
        output[slices_dst] = img[slices_src]
        return output
    
    qsm = center_crop_or_pad(qsm, input_shape)
    t1 = center_crop_or_pad(t1, input_shape)
    aal_mask = center_crop_or_pad(aal_mask, input_shape)
    
    # 堆叠成3通道输入
    patient_volume = np.stack([qsm, t1, aal_mask], axis=0)  # (C=3, D, H, W)
    
    # 检查是否需要翻转维度
    if patient_volume.shape[0] != 3:
        patient_volume = patient_volume.transpose(2, 0, 1)  # 确保(C, D, H, W)
    
    print(f"  预处理后: {patient_volume.shape}")
    
    # ========== Step 3: 加载模型 ==========
    print("\n[Step 3] 加载模型...")
    
    from models_patch import ResNet3D_PatchClassifier, PatchMILAggregator
    
    # 创建模型
    model = ResNet3D_PatchClassifier(
        in_channels=3,
        base_channels=cfg['model'].get('base_channels', 16),
        norm=cfg['model'].get('norm', 'group'),
        num_groups=cfg['model'].get('num_groups', 4),
        dropout=cfg['model'].get('dropout', 0.3)
    )
    
    # 加载权重
    checkpoint = torch.load(model_ckpt, map_location=device)
    
    # 处理不同的checkpoint格式
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除'module.'前缀（如果有）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"  模型已加载到 {device}")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== Step 4: 执行分层MIL管道 ==========
    print("\n[Step 4] 执行分层MIL管道...")
    
    patch_size = tuple(cfg['train'].get('patch_size', [48, 48, 32]))
    patch_stride = tuple(cfg['train'].get('patch_stride', patch_size))
    
    pooling = cfg['train'].get('pooling', 'topk_mean')
    topk_percent = cfg['train'].get('topk_percent', 0.15)

    results = complete_overlap_mil_pipeline(
        model=model,
        patient_volume=patient_volume,
        aal_mask=aal_mask,
        patch_size=patch_size,
        patch_stride=patch_stride,
        device=device,
        verbose=True,
        pooling=pooling,
        topk_percent=topk_percent
    )
    
    # ========== Step 5: 可视化和保存结果 ==========
    print("\n[Step 5] 保存结果...")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存数值结果
    np.savez(
        os.path.join(out_dir, 'inference_results.npz'),
        patient_prob=results['patient_prob'],
        voxel_heatmap=results['voxel_heatmap'],
        roi_scores=np.array(list(results['roi_scores'].values())),
        patch_logits=np.array(list(results['patch_logits'].values()))
    )
    
    # 保存热图为NIfTI格式（便于医学影像查看）
    try:
        import nibabel as nib
        
        # 保存体素级热图
        img_nii = nib.Nifti1Image(results['voxel_heatmap'], np.eye(4))
        nib.save(img_nii, os.path.join(out_dir, 'voxel_heatmap.nii.gz'))
        
        # 保存AAL掩码
        aal_nii = nib.Nifti1Image(aal_mask, np.eye(4))
        nib.save(aal_nii, os.path.join(out_dir, 'aal_mask.nii.gz'))
        
    except ImportError:
        pass
    
    # 保存JSON报告
    report = results['report']
    
    # 格式化报告
    json_report = {
        'patient_diagnosis': {
            'probability': report['patient_diagnosis'],
            'confidence': 'high' if report['patient_diagnosis'] > 0.7 else 
                         'medium' if report['patient_diagnosis'] > 0.3 else 'low'
        },
        'patch_statistics': report['patch_stats'],
        'voxel_coverage': report['voxel_coverage'],
        'top_10_rois': [
            {'roi_id': roi_id, 'score': score}
            for roi_id, score in report['top_rois']
        ],
        'summary': report['summary']
    }
    
    with open(os.path.join(out_dir, 'inference_report.json'), 'w') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ 结果保存到 {out_dir}")
    print(f"  ✓ 文件列表:")
    print(f"    - inference_results.npz (数值结果)")
    print(f"    - inference_report.json (分析报告)")
    print(f"    - voxel_heatmap.nii.gz (体素级热图)")
    print(f"    - aal_mask.nii.gz (脑区掩码)")
    
    # ========== Step 6: 打印详细分析 ==========
    print("\n[Step 6] 详细分析结果")
    print("-" * 70)
    
    print(f"\n【患者诊断】")
    print(f"  诊断概率: {report['patient_diagnosis']:.3f}")
    if report['patient_diagnosis'] > 0.7:
        print(f"  → 高风险 (概率 > 0.7)")
    elif report['patient_diagnosis'] > 0.3:
        print(f"  → 中等风险 (0.3 < 概率 < 0.7)")
    else:
        print(f"  → 低风险 (概率 < 0.3)")
    
    print(f"\n【Patch分析】")
    print(f"  总Patch数: {report['patch_stats']['num_patches']}")
    print(f"  平均Patch分数: {report['patch_stats']['mean_prob']:.3f}")
    print(f"  高风险Patch比例: {report['patch_stats']['high_prob_ratio']:.1%}")
    
    print(f"\n【Top-5 高风险Patch】")
    for i, (patch_coords, prob, roi_id) in enumerate(results['top_patches'][:5], 1):
        h, w, d = patch_coords
        print(f"  {i}. Patch({h:3d}, {w:3d}, {d:2d}) "
              f"概率={prob:.3f} "
              f"所在ROI=#{roi_id}")
    
    print(f"\n【Top-10 高风险脑区】")
    for i, (roi_id, score) in enumerate(report['top_rois'][:10], 1):
        print(f"  {i:2d}. ROI #{roi_id:3d}: {score:.3f}")
    
    print("\n" + "="*70)
    
    return results


# ============================================================================
# 简化的快速推理接口
# ============================================================================

def quick_inference(model_ckpt, qsm_path, t1_path, aal_path, out_dir, device='cuda'):
    """
    快速推理接口：只需提供必要的参数
    
    使用default config
    """
    
    # Default config
    cfg = {
        'data': {'input_size': [192, 192, 128]},
        'model': {
            'base_channels': 16,
            'norm': 'group',
            'num_groups': 4,
            'dropout': 0.3
        },
        'train': {
            'patch_size': [48, 48, 32],
            'patch_stride': [24, 24, 16],  # 50% overlap
            'device': device
        }
    }
    
    return run_inference_with_overlap_support(
        model_ckpt=model_ckpt,
        qsm_path=qsm_path,
        t1_path=t1_path,
        aal_path=aal_path,
        cfg=cfg,
        out_dir=out_dir,
        device=device
    )


# ============================================================================
# 与原有inference.py的集成建议
# ============================================================================

"""
如何在现有inference.py中集成:

1. 添加导入:
   from mil_overlap_utils import complete_overlap_mil_pipeline

2. 在推理函数中替换MIL聚合部分:
   
   原有方式:
   --------
   patch_logits = [model(p) for p in patches]
   patient_logit = topk_mean(patch_logits)
   
   新方式 (支持重叠):
   --------
   results = complete_overlap_mil_pipeline(
       model=model,
       patient_volume=patient_volume,
       aal_mask=aal_mask,
       patch_size=cfg['train']['patch_size'],
       patch_stride=cfg['train']['patch_stride'],
       device=device
   )
   patient_prob = results['patient_prob']
   voxel_heatmap = results['voxel_heatmap']

3. 处理config.yaml:
   
   检查patch_stride是否小于patch_size:
   - 如果 stride < size → 自动使用分层MIL
   - 如果 stride == size → 保持原有逻辑（后向兼容）

4. 向后兼容:
   
   if patch_stride == patch_size:
       # 无重叠: 使用原有快速路径
       patient_prob = original_topk_mean(patch_logits)
   else:
       # 有重叠: 使用新的分层MIL
       results = complete_overlap_mil_pipeline(...)
       patient_prob = results['patient_prob']
"""

if __name__ == '__main__':
    # 使用示例
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='模型checkpoint')
    parser.add_argument('--qsm', type=str, required=True, help='QSM脑图像')
    parser.add_argument('--t1', type=str, required=True, help='T1脑图像')
    parser.add_argument('--aal', type=str, required=True, help='AAL脑图谱')
    parser.add_argument('--out', type=str, required=True, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='cuda或cpu')
    
    args = parser.parse_args()
    
    results = quick_inference(
        model_ckpt=args.ckpt,
        qsm_path=args.qsm,
        t1_path=args.t1,
        aal_path=args.aal,
        out_dir=args.out,
        device=args.device
    )
