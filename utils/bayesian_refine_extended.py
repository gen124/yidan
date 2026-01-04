import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions import Normal, Beta
from torch.optim import Adam
import copy

# 导入原函数以保持兼容性（使用相对导入，避免包导入问题）
from .bayesian_refine import (
    subject_conditional_soft_refine,
    spatial_soft_refine,
    build_patch_neighbors,
    mil_pooling_from_patch_scores
)

def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# ===== 扩展Bayesian框架：变分推断 (Variational Inference) =====

class VariationalPatchRefiner(nn.Module):
    """
    使用变分推断对patch概率进行贝叶斯更新。
    假设patch_label ~ Beta(alpha, beta)，使用VI近似后验。
    """
    def __init__(self, n_patches, hidden_dim=32):
        super().__init__()
        self.n_patches = n_patches
        # 变分参数：alpha和beta的对数（确保正值）
        self.log_alpha = nn.Parameter(torch.randn(n_patches))
        self.log_beta = nn.Parameter(torch.randn(n_patches))

    def forward(self, subject_prob, spatial_context=None):
        """
        subject_prob: scalar, subject-level probability
        spatial_context: optional, tensor of shape (n_patches,) for spatial priors
        Returns: sampled probabilities from variational posterior
        """
        alpha = F.softplus(self.log_alpha) + 1e-3  # alpha > 0
        beta_param = F.softplus(self.log_beta) + 1e-3  # beta > 0

        # 融入subject prior: 调整alpha/beta基于subject_prob
        subject_strength = 2.0  # 超参数
        alpha_adj = alpha + subject_strength * subject_prob
        beta_adj = beta_param + subject_strength * (1 - subject_prob)

        if spatial_context is not None:
            # 融入spatial prior
            alpha_adj = alpha_adj + 0.5 * spatial_context
            beta_adj = beta_adj + 0.5 * (1 - spatial_context)

        # 从Beta分布采样
        dist = Beta(alpha_adj, beta_adj)
        samples = dist.rsample((10,))  # 采样10次以估计期望
        mean_prob = samples.mean(dim=0)
        uncertainty = samples.std(dim=0)  # 不确定性度量

        return mean_prob, uncertainty

    def kl_divergence(self, prior_alpha=1.0, prior_beta=1.0):
        """
        计算与先验的KL散度，用于VI损失
        """
        alpha = F.softplus(self.log_alpha) + 1e-3
        beta_param = F.softplus(self.log_beta) + 1e-3
        prior_dist = Beta(prior_alpha, prior_beta)
        posterior_dist = Beta(alpha, beta_param)
        return torch.distributions.kl_divergence(posterior_dist, prior_dist).sum()

def variational_subject_conditional_refine(patch_probs, subject_scores, n_epochs=50, lr=1e-2):
    """
    使用VI进行subject-conditional refine
    patch_probs: dict {(sid, pid): prob}
    subject_scores: dict {sid: prob}
    Returns: refined probs dict, uncertainties dict
    """
    # 按subject分组
    subject_groups = {}
    for (sid, pid), prob in patch_probs.items():
        if sid not in subject_groups:
            subject_groups[sid] = []
        subject_groups[sid].append((pid, prob))

    refined_probs = {}
    uncertainties = {}

    for sid, patches in subject_groups.items():
        n_patches = len(patches)
        subject_prob = subject_scores[sid]

        # 初始化VI模型
        vi_model = VariationalPatchRefiner(n_patches)
        optimizer = Adam(vi_model.parameters(), lr=lr)

        # VI训练
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            mean_probs, uncerts = vi_model(subject_prob)
            # 简单重建损失：最小化与观测prob的MSE
            observed = torch.tensor([p for _, p in patches])
            loss = F.mse_loss(mean_probs, observed) + 0.1 * vi_model.kl_divergence()
            loss.backward()
            optimizer.step()

        # 最终采样
        final_probs, final_uncerts = vi_model(subject_prob)
        for i, (pid, _) in enumerate(patches):
            refined_probs[(sid, pid)] = final_probs[i].item()
            uncertainties[(sid, pid)] = final_uncerts[i].item()

    return refined_probs, uncertainties

# ===== MCMC采样 (MCMC Sampling) =====

def mcmc_patch_refine(patch_probs, subject_scores, n_samples=1000, burn_in=200):
    """
    使用Metropolis-Hastings MCMC采样patch后验分布
    """
    refined_probs = {}
    uncertainties = {}

    for (sid, pid), initial_prob in patch_probs.items():
        subject_prob = subject_scores[sid]

        # 提议分布：正态分布
        proposal_std = 0.1
        current = initial_prob
        samples = []

        for _ in range(n_samples + burn_in):
            # 提议新值
            proposed = np.clip(current + np.random.normal(0, proposal_std), 0, 1)

            # 计算接受率 (简化：基于subject prior)
            log_accept = (proposed - current) * (2 * subject_prob - 1)  # 偏向subject_prob
            if np.log(np.random.rand()) < log_accept:
                current = proposed
            if _ >= burn_in:
                samples.append(current)

        mean_prob = np.mean(samples)
        uncertainty = np.std(samples)
        refined_probs[(sid, pid)] = mean_prob
        uncertainties[(sid, pid)] = uncertainty

    return refined_probs, uncertainties

# ===== 算法优化 =====

def adaptive_hyperparameters(patch_probs, subject_scores, val_patch_probs, val_subject_scores, candidates=None):
    """
    使用验证集自适应选择超参数 (alpha, lambda_spatial)
    candidates: dict of hyperparam combinations
    Returns: best hyperparams
    """
    if candidates is None:
        candidates = [
            {'alpha': 1.0, 'lambda_spatial': 0.1},
            {'alpha': 1.5, 'lambda_spatial': 0.3},
            {'alpha': 2.0, 'lambda_spatial': 0.5},
        ]

    best_score = -np.inf
    best_params = None

    # 转换格式为subject_conditional_soft_refine期望的格式
    train_patch_dict = convert_patch_scores_to_dict(patch_probs)

    for params in candidates:
        # 训练集refine - subject_conditional_soft_refine返回的是flat字典格式
        refined_train_flat, _ = subject_conditional_soft_refine(train_patch_dict, subject_scores, alpha=params['alpha'])

        # 应用spatial refine
        refined_train_flat = spatial_soft_refine(refined_train_flat, {}, lambda_spatial=params['lambda_spatial'], n_iters=2)

        # 验证集评估 (简化：计算与val的差异)
        val_diff = 0
        count = 0
        for key in val_patch_probs:
            if key in refined_train_flat:
                val_diff += abs(refined_train_flat[key] - val_patch_probs[key])
                count += 1

        if count > 0:
            score = -val_diff / count  # 负MAE
        else:
            score = -np.inf

        if score > best_score:
            best_score = score
            best_params = params

    return best_params

# ===== 多尺度建模 =====

def multi_scale_refine(patch_probs, slice_probs=None, volume_prob=None, alpha=1.5):
    """
    多尺度refine：融入slice和volume级信息
    patch_probs: dict {(sid, patch_idx): prob}
    slice_probs: dict {slice_id: prob} (暂时未使用)
    volume_prob: scalar
    """
    refined = copy.deepcopy(patch_probs)

    # 当前简化实现：只使用volume级prior
    for key, prob in patch_probs.items():
        # key格式: (sid, patch_idx)
        volume_prior = volume_prob if volume_prob is not None else 0.5

        # 简单组合：当前prob和volume prior的加权平均
        combined_prior = (prob + volume_prior) / 2
        refined[key] = sigmoid(logit(prob) + alpha * logit(combined_prior))

    return refined

# ===== 鲁棒性提升 =====

def robust_spatial_soft_refine(patch_probs, patch_neighbors, lambda_spatial=0.3, n_iters=2, noise_std=0.05, mc_dropout=0.1):
    """
    增强鲁棒性的spatial refine：噪声注入 + MC Dropout
    """
    refined = copy.deepcopy(patch_probs)

    for _ in range(n_iters):
        new_refined = {}
        for key, prob in refined.items():
            # 噪声注入
            noisy_prob = np.clip(prob + np.random.normal(0, noise_std), 0, 1)

            # MC Dropout模拟
            if np.random.rand() < mc_dropout:
                noisy_prob = 0.5  # dropout to prior

            # 邻域平均
            neighbors = patch_neighbors.get(key, [])
            if neighbors:
                neighbor_probs = [refined.get(n, noisy_prob) for n in neighbors]
                neighbor_avg = np.mean(neighbor_probs)
                updated = (1 - lambda_spatial) * noisy_prob + lambda_spatial * neighbor_avg
            else:
                updated = noisy_prob

            new_refined[key] = updated
        refined = new_refined

    return refined

# ===== 保持原有函数以兼容 =====

def subject_conditional_soft_refine(patch_scores_dict, subject_scores, alpha=1.5, freeze_high_conf=True, high_conf_threshold=0.8, low_conf_threshold=0.2):
    """
    原有函数，保持不变 - 直接调用原函数
    """
    from .bayesian_refine import subject_conditional_soft_refine as orig_func
    return orig_func(patch_scores_dict, subject_scores, alpha, freeze_high_conf, high_conf_threshold, low_conf_threshold)

def spatial_soft_refine(patch_probs, patch_neighbors, lambda_spatial=0.3, n_iters=2):
    """
    原有函数，保持不变 - 直接调用原函数
    """
    from .bayesian_refine import spatial_soft_refine as orig_func
    return orig_func(patch_probs, patch_neighbors, lambda_spatial, n_iters)

def convert_patch_scores_to_dict(patch_scores):
    """Convert patch scores dict to per-subject dict"""
    result = {}
    for (sid, pidx), score in patch_scores.items():
        if sid not in result:
            result[sid] = []
        result[sid].append(score)
    return result

def build_patch_neighbors(patch_index_map):
    """Build patch neighbors from index map"""
    # 简化实现，实际应从原函数复制
    neighbors = {}
    for key in patch_index_map:
        neighbors[key] = []  # 空邻域，简化版
    return neighbors

def mil_pooling_from_patch_scores(patch_scores, k_ratio=0.15):
    """MIL pooling from patch scores - direct implementation"""
    subject_pools = {}
    for (sid, _), score in patch_scores.items():
        if sid not in subject_pools:
            subject_pools[sid] = []
        subject_pools[sid].append(score)

    subject_scores = {}
    for sid, scores in subject_pools.items():
        scores_tensor = torch.tensor(scores)
        if len(scores) > 1:
            k = max(1, int(len(scores) * k_ratio))
            topk_vals = torch.topk(scores_tensor, k)[0]
            subject_scores[sid] = topk_vals.mean().item()
        else:
            subject_scores[sid] = scores[0]

    return subject_scores

