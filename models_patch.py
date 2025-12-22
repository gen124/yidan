"""
简化版3D ResNet用于Patch级分类(移除GNN,专注MIL)

核心改动:
1. 移除Patch-GNN相关代码
2. 保留ResNet骨干 + Patch分类头
3. 支持冻结/解冻骨干的分阶段训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, norm='group', num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if norm == 'group':
            self.bn1 = nn.GroupNorm(num_groups, planes)
        else:
            self.bn1 = nn.BatchNorm3d(planes)
        
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == 'group':
            self.bn2 = nn.GroupNorm(num_groups, planes)
        else:
            self.bn2 = nn.BatchNorm3d(planes)
        
        self.downsample = None
        if stride != 1 or in_planes != planes:
            if norm == 'group':
                self.downsample = nn.Sequential(
                    nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(num_groups, planes)
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes)
                )
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = F.relu(out)
        return out


class ResNet3D_PatchClassifier(nn.Module):
    """
    3D ResNet用于Patch级二分类
    
    特性:
    - 输入: 单个3D patch (C, D, H, W)
    - 输出: patch级分类logit (标量)
    - 支持冻结骨干进行分阶段训练
    """
    def __init__(self, in_channels=3, base_channels=16, block=BasicBlock3D, 
                 layers=[2,2,2,2], num_classes=1, norm='group', num_groups=8, dropout=0.3):
        super().__init__()
        self.in_planes = base_channels
        self.norm = norm
        self.num_groups = num_groups
        
        # 骨干网络
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=(1,2,2), padding=3, bias=False)
        if norm == 'group':
            self.bn1 = nn.GroupNorm(num_groups, base_channels)
        else:
            self.bn1 = nn.BatchNorm3d(base_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels*2, layers[1], stride=(1,2,2))
        self.layer3 = self._make_layer(block, base_channels*4, layers[2], stride=(1,2,2))
        self.layer4 = self._make_layer(block, base_channels*8, layers[3], stride=1)
        
        # 全局池化和分类头
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s, norm=self.norm, num_groups=self.num_groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, C, D, H, W) 输入patch
            return_features: 是否返回池化后的特征向量
        
        Returns:
            logits: (B,) patch级分类分数
            features: (B, feat_dim) 可选,用于后续聚合
        """
        # 特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局池化
        pooled = self.avgpool(x)
        pooled = pooled.view(pooled.size(0), -1)
        
        if self.dropout is not None:
            pooled = self.dropout(pooled)
        
        # 分类
        logits = self.fc(pooled).squeeze(-1)
        
        if return_features:
            return logits, pooled
        return logits
    
    def freeze_backbone(self):
        """冻结骨干网络参数(仅训练分类头)"""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters():
                param.requires_grad = False
        print("[INFO] 骨干网络已冻结,仅训练分类头")
    
    def unfreeze_backbone(self):
        """解冻骨干网络参数(联合训练)"""
        for param in self.parameters():
            param.requires_grad = True
        print("[INFO] 骨干网络已解冻,联合训练")


class PatchMILAggregator(nn.Module):
    """
    将多个patch的预测聚合为患者级决策
    
    支持的聚合方式:
    - max: 最大池化(MIL标准假设)
    - topk_mean: 取前k%高分patch的均值
    - attention: 学习注意力权重加权求和
    """
    def __init__(self, feat_dim=128, pooling='topk_mean', topk_percent=0.15):
        super().__init__()
        self.pooling = pooling
        self.topk_percent = topk_percent
        
        if pooling == 'attention':
            self.attn_fc = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1)
            )
    
    def forward(self, patch_logits, patch_features=None):
        """
        Args:
            patch_logits: (B, N) 或 (N,) patch级logits
            patch_features: (B, N, D) 可选,patch特征向量
        
        Returns:
            patient_logit: (B,) 患者级聚合分数
        """
        if patch_logits.dim() == 1:
            patch_logits = patch_logits.unsqueeze(0)  # (1, N)
        
        B, N = patch_logits.shape
        
        if self.pooling == 'max':
            patient_logit = patch_logits.max(dim=1)[0]
        
        elif self.pooling == 'topk_mean':
            k = max(1, int(N * self.topk_percent))
            topk_vals = torch.topk(patch_logits, k, dim=1)[0]
            patient_logit = topk_vals.mean(dim=1)
        
        elif self.pooling == 'attention':
            if patch_features is None:
                raise ValueError("attention pooling需要patch_features")
            # (B, N, D) -> (B, N, 1)
            attn_scores = self.attn_fc(patch_features).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            patient_logit = (patch_logits * attn_weights).sum(dim=1)
        
        else:
            raise ValueError(f"未知的pooling方式: {self.pooling}")
        
        return patient_logit


def build_patch_model(cfg):
    """从config构建Patch分类模型"""
    model_cfg = cfg['model']
    return ResNet3D_PatchClassifier(
        in_channels=model_cfg.get('in_channels', 3),
        base_channels=model_cfg.get('base_channels', 16),
        layers=model_cfg.get('layers', [2,2,2,2]),
        num_classes=1,
        norm=model_cfg.get('norm', 'group'),
        num_groups=model_cfg.get('num_groups', 8),
        dropout=model_cfg.get('dropout', 0.3)
    )
