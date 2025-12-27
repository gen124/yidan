"""
3D ResNet用于Patch级分类

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
    def __init__(self, feat_dim=128, pooling='topk_mean', topk_percent=0.15, tau=0.1, q=0.9, q_min=0.5, q_ref_n=100):
        super().__init__()
        self.pooling = pooling
        self.topk_percent = topk_percent
        self.tau = tau
        self.q = q
        # adaptive quantile params
        self.q_min = q_min
        self.q_ref_n = q_ref_n

        # unified head: project pooling output to `feat_dim`, then scalar head
        if pooling == 'distribution':
            self.input_proj = nn.Linear(feat_dim * 3, feat_dim)
        else:
            self.input_proj = nn.Identity()
        self.subject_head = nn.Linear(feat_dim, 1)

        # attention module operates on features
        if pooling == 'attention':
            self.attn_fc = nn.Sequential(
                nn.Linear(feat_dim, max(8, feat_dim // 4)),
                nn.ReLU(),
                nn.Linear(max(8, feat_dim // 4), 1)
            )

    def forward(self, patch_logits, patch_features=None):
        """
        Args:
            patch_logits: (B, N) 或 (N,) 可选的 patch logits/scores
            patch_features: (B, N, D) 可选的 patch features

        Returns:
            patient_logit: (B,) 患者级logit
        """
        # normalize dims
        if patch_logits is not None and patch_logits.dim() == 1:
            patch_logits = patch_logits.unsqueeze(0)  # (1, N)

        B = None
        N = None
        if patch_features is not None:
            if patch_features.dim() == 2:
                patch_features = patch_features.unsqueeze(0)
            B, N, D = patch_features.shape
        elif patch_logits is not None:
            if patch_logits.dim() == 1:
                patch_logits = patch_logits.unsqueeze(0)
            B, N = patch_logits.shape

        # If using legacy logit-based pooling
        if self.pooling in ['max', 'topk_mean', 'mean']:
            if patch_logits is None:
                raise ValueError(f"pooling '{self.pooling}' 需要 patch_logits")
            if self.pooling == 'max':
                return patch_logits.max(dim=1)[0]
            elif self.pooling == 'topk_mean':
                k = max(1, int(N * self.topk_percent))
                topk_vals = torch.topk(patch_logits, k, dim=1)[0]
                return topk_vals.mean(dim=1)
            else:  # mean
                return patch_logits.mean(dim=1)

        # Feature-based pooling (主路径)
        if patch_features is None:
            raise ValueError(f"pooling '{self.pooling}' 需要 patch_features")

        # scores for soft selection: prefer provided logits, else use L2 norm of features
        if patch_logits is not None:
            scores = patch_logits.squeeze(0) if patch_logits.dim() == 2 and patch_logits.shape[0] == 1 else patch_logits
            # ensure shape (B,N)
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)
        else:
            # compute a proxy score from features
            scores = patch_features.norm(p=2, dim=2)  # (B, N)

        if self.pooling == 'distribution':
            # compute per-batch pooled feature
            # Normalize features (L2 per-patch) before distribution pooling to remove scale differences
            eps = 1e-8
            f_norm = patch_features.norm(p=2, dim=2, keepdim=True).clamp_min(eps)
            f_normed = patch_features / f_norm
            mu = f_normed.mean(dim=1)        # (B, D)
            var = f_normed.var(dim=1, unbiased=False)  # (B, D)
            mx, _ = f_normed.max(dim=1)      # (B, D)
            z = torch.cat([mu, var, mx], dim=1)    # (B, 3D)
            z = self.input_proj(z)                 # (B, D)
            logit = self.subject_head(z).squeeze(-1)
            return logit

        elif self.pooling == 'soft_topk':
            # softmax over scores / tau -> weighted sum of features
            w = torch.softmax(scores / float(self.tau), dim=1)  # (B, N)
            w = w.unsqueeze(-1)  # (B, N, 1)
            z = (w * patch_features).sum(dim=1)  # (B, D)
            z = self.input_proj(z)
            logit = self.subject_head(z).squeeze(-1)
            return logit

        elif self.pooling == 'quantile':
            # select patches with score >= adaptive quantile threshold, average their features
            # compute adaptive q based on patch count N: when N < q_ref_n, scale q down proportionally
            # q_adj = max(q_min, q_base * min(1.0, N / q_ref_n))
            N_curr = N if isinstance(N, int) else int(N)
            q_base = float(self.q)
            q_adj = max(float(self.q_min), q_base * min(1.0, float(N_curr) / float(self.q_ref_n)))
            thresh = torch.quantile(scores, q_adj, dim=1, keepdim=True)  # (B,1)
            mask = (scores >= thresh)  # (B, N)
            # avoid empty mask: if none selected, fallback to mean
            masked_feats = []
            for b in range(B):
                m = mask[b]
                if m.any():
                    sel = patch_features[b][m]
                    masked_feats.append(sel.mean(dim=0))
                else:
                    masked_feats.append(patch_features[b].mean(dim=0))
            z = torch.stack(masked_feats, dim=0)  # (B, D)
            z = self.input_proj(z)
            logit = self.subject_head(z).squeeze(-1)
            return logit

        elif self.pooling == 'attention':
            # compute attention scores from features
            attn_scores = self.attn_fc(patch_features).squeeze(-1)  # (B, N)
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, N, 1)
            z = (attn_weights * patch_features).sum(dim=1)  # (B, D)
            z = self.input_proj(z)
            logit = self.subject_head(z).squeeze(-1)
            return logit

        else:
            raise ValueError(f"未知的pooling方式: {self.pooling}")


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
