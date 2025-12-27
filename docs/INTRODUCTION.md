# PD-3D-Graph-Explainer — 项目介绍

## 概述

PD-3D-Graph-Explainer 是一个面向 3D 医学影像（QSM/T1 + ROI segmentation）的弱监督分类与可解释性研究工具箱。

核心思想：以 3D patch 为基本单元训练 ResNet3D 编码器与 patch 级分类器，通过 Patch-MIL 聚合得到患者级预测；解释阶段把 patch 级证据投回到 voxel/ROI 空间，并用轻量 Tiny-GNN 提供结构化后验解释。

## 设计原则与亮点

- 忠实性（Faithfulness）：解释基于训练得到的 attention head 与模型输出，禁止使用未训练随机 attention 进行解释。
- 重叠友好（Overlap-aware）：利用重叠 patch 的多次覆盖作为“多次投票”生成平滑、鲁棒的 voxel-level heatmap。
- 结构化解释：在 ROI / supervoxel 级别构图，训练 Tiny-GNN（post-hoc）作为 surrogate explainer，输出节点/边重要性与证据路径。
- 模块化：分离数据加载、patch 模型、聚合器、图构建与解释器，便于维护与扩展。

## 项目结构（核心文件）

- `models_patch.py` — Patch 分类器（`ResNet3D_PatchClassifier`）与 `PatchMILAggregator`（支持 `max` / `topk_mean` / `attention`）。
- `data_utils.py` — NIfTI 读写、z-score 归一化、center-crop/pad、`PDVolDataset` / `PatchDataset`。
- `scripts/explain_attention_overlap.py` — 解释入口脚本：滑窗推理 → attention 计算 → heatmap 投票 → ROI 图构建 → Tiny-GNN 解释（当前为自包含脚本）。
- `graph_builder.py` — 将 voxel/heatmap 聚合为 ROI / slice / supervoxel 图（输出 nodes, feats, edges）。
- `explainer.py` — TinyGNN（基于 GAT）与解释训练/解释逻辑（`fit_tiny_gnn_and_explain`、`path_pdm`）。
- `inference.py` — 包含常用推理工具（例如 `center_crop_or_pad`、Grad-CAM）；建议将批量 patch 抽取接口 `extract_patch_features()` 放入此处以便复用。

> 注：原先的 `mil_overlap_utils.py` 中的最小 helper 已合并到解释脚本；如果需要完整的 overlap 管道，可将其恢复为独立模块。

## 方法学细节（用于论文 Methods）

### 两类热力图语义

必须在论文中区分两类热力图，并在实验中报告两者对比：

1. **Importance map（attention-only）**：
   \[ H(v) = \frac{1}{|P(v)|} \sum_{i\in P(v)} \alpha_i \]
   表示模型“关注”强度（attention）。

2. **Evidence map（attention × prob）**：
   \[ E(v) = \frac{1}{|P(v)|} \sum_{i\in P(v)} \alpha_i \cdot \sigma(z_i) \]
   其中 \(z_i\) 为 patch logit，\(\sigma\) 为 sigmoid。此映射反映 patch 对疾病判定的支持程度，更接近模型输出。

### 空间权重

对每个 patch 内部可以采用中心加权（linear / cosine kernel）以降低边缘不确定性引入的伪阳性。

### Tiny-GNN 训练目标（可选）

- 方案 A（推荐）：训练 Tiny-GNN 去拟合主模型的 `patient_prob`，作为 surrogate model，保证解释的忠实性。
- 方案 B：训练 Tiny-GNN 去拟合 attention 分布，用于研究 attention 的结构化模式。

建议对 small-sample regime 采用多次训练并取平均（`n_runs >= 3`）以提高解释稳定性。

## 快速上手

### 环境

```bash
conda create -n patchmil-gpu python=3.10 -y
conda activate patchmil-gpu
pip install -r requirements.txt
```

### 训练（示例）

```bash
PYTHONUNBUFFERED=1 python -u train.py --config config.yaml --out_dir outputs/exp4
```

### 单患者解释（端到端示例）

```bash
python scripts/explain_attention_overlap.py \
  --ckpt outputs/exp4/final_best.pth \
  --agg_ckpt outputs/exp4/aggregator_best.pth \
  --qsm data/PD/PD001/qsm.nii \
  --t1 data/PD/PD001/t1.nii \
  --aal data/PD/PD001/aal.nii \
  --cfg config.yaml \
  --out_dir outputs/explain/PD001 \
  --device cuda \
  --heatmap_mode att_prob \
  --edge_weight_method cosine
```

**说明**：`--agg_ckpt` 必需；`--heatmap_mode` 支持 `att_prob`（默认）或 `att_only`；`--edge_weight_method` 支持 `cosine`、`linear`、`uniform`。

### 保存热力图为 NIfTI（示例）

```python
import numpy as np, nibabel as nib
arr = np.load('outputs/explain/PD001/voxel_importances.npz')['voxel_probs']
img = nib.Nifti1Image(arr.astype('float32'), np.eye(4))
nib.save(img, 'outputs/explain/PD001/voxel_heatmap.nii.gz')
```

## 输出文件说明

- `outputs/explain/<pid>/voxel_importances.npz` — 包含 `voxel_probs`、`voxel_votes`、`voxel_weights`。
- `outputs/explain/<pid>/explain_summary.json` — 概要信息（`patient_prob`、Top-K patches、GNN node importance 等）。
- （可选）Top-K patch crops、NIfTI 热力图用于论文图示。

## 复现实验要点与建议

- 记录并保存完整 `config.yaml`、训练日志、patch 模型 checkpoint 与 aggregator checkpoint。
- 固定随机种子（`torch.manual_seed`、`np.random.seed`）与记录硬件信息以便复现。
- 对 Tiny-GNN 使用 `n_runs>=3` 并平均结果以减小单次训练波动。
- 对比 `att_only` 与 `att_prob` 两种热力图，报告定位一致性与判别性差异。

## 常见问题与调试

- **缺少 aggregator**：解释必须基于训练好的 attention aggregator，请通过 `--agg_ckpt` 提供；脚本会校验并拒绝使用未训练 head。
- **checkpoint 加载失败**：确认 checkpoint 是 `state_dict` 或包含 `ck['state_dict']`，并检查是否为 DataParallel 保存（脚本尝试处理 `module.` 前缀）。
- **热力图噪点/伪阳性**：尝试 `edge_weight_method=cosine` 或使用 `att_only` 做对照；增加 patch 内部中心权重。
- **GNN 解释不稳定**：增大 `n_runs`，减小 `noise_std`，或为节点增加更多 ROI-level 特征。

## 局限与后续工作

- 推理速度：patch-based 推理在高清体积上较慢，建议批量化（`PatchDataset` + `DataLoader`）或研究全卷积替代方案。
- 解释因果性：Tiny-GNN 为后验 surrogate，解释受训练目标与数据分布限制，需与 Grad-CAM 等方法互证。
- 后续改进：将 `extract_patch_features()` 集中到 `inference.py`、扩展 ROI 特征、实现对解释结果的定量评估（IoU/AUPR）等。

## 贡献与引用

欢迎 issue / PR。若在论文中使用本方法，请在 Methods 中说明 attention→voxel heatmap 与 Tiny-GNN 的解释流程，并引用本仓库。

---

如需我把此文档再压缩为 1 页的 Supplement 或直接生成 Figures 说明文字，我可以继续处理。