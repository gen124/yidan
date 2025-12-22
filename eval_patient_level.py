import torch
import yaml
import numpy as np
import os
from torch.utils.data import DataLoader
from data_utils import PDVolDataset, PatchDataset
from models_patch import build_patch_model, PatchMILAggregator
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from tqdm import tqdm

def eval_patient_level(exp_dir='outputs/exp2', config_path='config.yaml'):
    # 1. 加载配置
    # 优先使用实验目录下的 config.yaml
    exp_config = os.path.join(exp_dir, 'config.yaml')
    if os.path.exists(exp_config):
        print(f"[INFO] 使用实验目录下的配置: {exp_config}")
        config_path = exp_config
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    # 2. 加载模型
    model = build_patch_model(cfg).to(device)
    ckpt_path = os.path.join(exp_dir, 'final_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] 找不到模型文件: {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device)
    # 处理 DataParallel 包装
    if all(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.eval()

    # 3. 加载验证集
    val_dataset = PDVolDataset(
        manifest_csv=cfg['data']['val_csv'],
        input_size=cfg['data']['input_size'],
        mode='val'
    )
    patch_val_dataset = PatchDataset(
        base_dataset=val_dataset,
        patch_size=tuple(cfg['train']['patch_size']),
        stride=tuple(cfg['train']['patch_stride']),
        mode='pretrain'
    )
    
    print(f"[INFO] 验证集患者数: {len(val_dataset)}")
    print(f"[INFO] 验证集总Patch数: {len(patch_val_dataset)}")

    # 4. 推理
    patient_results = {} # {pid: {'logits': [], 'features': [], 'label': label}}
    
    # 使用较大的 batch_size 加速
    loader = DataLoader(patch_val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    pooling_type = cfg['train'].get('pooling', 'topk_mean')
    need_features = (pooling_type == 'attention')

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
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

    # 5. 聚合
    # 获取特征维度
    feat_dim = 0
    if need_features and len(patient_results) > 0:
        first_pid = list(patient_results.keys())[0]
        if len(patient_results[first_pid]['features']) > 0:
            feat_dim = patient_results[first_pid]['features'][0].shape[0]

    aggregator = PatchMILAggregator(
        feat_dim=feat_dim,
        pooling=pooling_type,
        topk_percent=cfg['train'].get('topk_percent', 0.15)
    ).to(device)
    
    # 如果是 attention，需要加载 aggregator 的权重 (如果有的话)
    # 注意：在当前代码中，attn_fc 是在 ResNet3D_PatchClassifier 之外的？
    # 不对，PatchMILAggregator 是独立的。在 train.py 中它是如何训练的？
    # 检查 train.py 发现它并没有显式训练 PatchMILAggregator... 
    # 难道 train.py 只训练了 PatchClassifier？
    
    y_true = []
    y_prob = []

    for pid, data in patient_results.items():
        logits = torch.tensor(data['logits']).to(device).unsqueeze(0) # (1, N)
        features = torch.tensor(np.array(data['features'])).to(device).unsqueeze(0) if need_features else None # (1, N, D)
        
        # Aggregator expects (B, N) or (N,)
        # 这里我们只有一个患者，所以传 (1, N) 和 (1, N, D)
        patient_logit = aggregator(logits, features)
        prob = torch.sigmoid(patient_logit).item()
        
        y_true.append(data['label'])
        y_prob.append(prob)

    # 6. 计算指标
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # 寻找最佳阈值 (Youden's Index: J = TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    # 使用默认阈值 0.5 的结果
    y_pred_05 = (y_prob >= 0.5).astype(int)
    acc_05 = accuracy_score(y_true, y_pred_05)
    
    # 使用最佳阈值的结果
    y_pred_best = (y_prob >= best_threshold).astype(int)
    acc_best = accuracy_score(y_true, y_pred_best)
    
    auc = roc_auc_score(y_true, y_prob)
    
    print("\n" + "="*30)
    print(f"患者级评估结果 ({exp_dir})")
    print(f"聚合方式: {pooling_type}")
    print(f"患者总数: {len(y_true)}")
    print(f"正样本数: {sum(y_true)}")
    print(f"负样本数: {len(y_true) - sum(y_true)}")
    print("-" * 30)
    print(f"患者级 AUC: {auc:.4f}")
    print(f"默认阈值 (0.5) ACC: {acc_05:.4f}")
    print(f"最佳阈值 ({best_threshold:.4f}) ACC: {acc_best:.4f}")
    print("="*30)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='outputs/exp2')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    eval_patient_level(args.exp_dir, args.config)
