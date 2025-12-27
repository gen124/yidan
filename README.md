# 🧠 PD-3D-Graph-Explainer
"""
PD-3D-Graph-Explainer
=================================

PD-3D-Graph-Explainer
=====================

概述
----
PD-3D-Graph-Explainer 是一个面向 3D 医学影像（QSM/T1 + ROI segmentation）的弱监督分类与可解释性研究工具箱。
核心思想：以 patch 为基本单元训练 3D ResNet 编码器与 patch 级分类器，通过 Patch-MIL 聚合得到患者级预测；解释阶段把 patch 级证据投回到 voxel/ROI 空间，并用小型 GNN 提供结构化解释。

项目亮点
--------
- 支持多种聚合策略：`max` / `topk_mean` / `attention`（stage5 可训练 attention aggregator）。
- Overlap-aware 解释：将重叠 patch 的 attention 或证据投票回 voxel，生成平滑、鲁棒的热力图。
- Tiny-GNN 后验解释器：把 ROI / supervoxel 作为节点，训练轻量 GNN 作为 surrogate explainer，输出节点与边的重要性分数。

核心文件（快速导航）
------------------
- `models_patch.py` — Patch 分类器（ResNet3D）与 `PatchMILAggregator`。
- `data_utils.py` — NIfTI 加载、z-score 归一化、center-crop/pad、`PDVolDataset`/`PatchDataset`。
- `scripts/explain_attention_overlap.py` — 解释入口：滑窗推理 → attention 计算 → heatmap 投票 → ROI 图构建 → Tiny-GNN 解释（为自包含单文件）。
- `graph_builder.py` — 将 voxel / heatmap 聚合为 ROI/slice/supervoxel 图（输出 nodes, feats, edges）。
- `explainer.py` — TinyGNN（GAT-based）与解释器训练/说明逻辑（`fit_tiny_gnn_and_explain`、`path_pdm`）。
- `inference.py` — 常用推理工具（建议将批量 patch 抽取接口 `extract_patch_features()` 放在此处以便复用）。

快速安装
-----------
建议使用 conda 创建隔离环境并安装依赖：

```bash
conda create -n patchmil-gpu python=3.10 -y
conda activate patchmil-gpu
pip install -r requirements.txt
```

数据约定
--------
- Manifest 行：`qsm_path,t1_path,aal_path,label,id`。
- 输入体积 shape：`(C=3, D, H, W)`（通道顺序 QSM, T1, AAL）。若 `aal` 缺失，代码会以 0 填充对应通道。

如何运行解释器（示例）
-----------------------
注意：解释必须使用训练好的 attention aggregator（`--agg_ckpt`）。

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

参数说明
----------
- `--agg_ckpt`：stage5 训练得到的 attention aggregator（必需，用于提取 attention 权重）。
- `--heatmap_mode`：`att_prob`（attention × sigmoid(logit)，证据热图）或 `att_only`（仅 attention 重要性）。
- `--edge_weight_method`：patch 内部的空间权重策略（`cosine|linear|uniform`），用于减少边界效应。

输出文件
--------
- `outputs/explain/<pid>/voxel_importances.npz`：`voxel_probs`、`voxel_votes`、`voxel_weights`（可转 NIfTI）。
- `outputs/explain/<pid>/explain_summary.json`：`patient_prob`（aggregator 计算）、Top-K patches、GNN 节点重要性等。




