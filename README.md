# 🧠 PD-3D-Graph-Explainer
"""
PD-3D-Graph-Explainer
=================================

项目目标
--------
一个针对小样本 3D 医学影像的可解释分类工具箱。核心流程：3D Patch 编码器 → Patch 级分类器 → Patch-MIL 聚合（topk_mean / attention / overlap-aware）→ 患者级决策；可选的 Grad-CAM + Tiny-GNN 后验解释器用于提供 voxel/ROI/supervoxel 级重要性。

目录快速浏览
----------------
- `data_utils.py`       数据加载、预处理与 `PDVolDataset` / `PatchDataset`。
- `models_patch.py`     `ResNet3D_PatchClassifier`, `PatchMILAggregator`。
- `train.py`            分阶段训练脚本（预训练、重标、微调、stage5 attention）。
- `inference.py`        推理 + Grad-CAM + 图构建 + explainer 编排。
- `explainer.py`        Tiny GNN explainer（GAT-based）和可视化工具。
- `graph_builder.py`    从 feature-map 构建 ROI / slice / supervoxel 图。
- `eval_patient_level.py` 患者级评估（支持加载 `aggregator_best.pth`）。
- `eval_patient_level_overlap.py`  重叠感知的 MIL 评估实现（替代聚合策略）。
- `scripts/`            包含对比与辅助脚本（例如 `compare_aggregators.py`）。

安装（建议）
----------------
推荐使用 conda：

```bash
conda create -n patchmil-gpu python=3.10 -y
conda activate patchmil-gpu
pip install -r requirements.txt
```

快速 smoke-test
```bash
# 合成/小样本测试，实时日志推荐无缓冲
PYTHONUNBUFFERED=1 python -u train.py --synthetic --out_dir outputs/test_smoke
```

数据格式
---------
- manifest 行格式：`qsm_path,t1_path,aal_path,label,id`。
- 输入体积形状（单例）：`(C=3, D, H, W)`（通道顺序 QSM, T1, AAL）；如果 `aal` 丢失，代码会用 0 填充以保留样本。
- Atlas/AAL 文件：项目并不强制某一具体 AAL 版本；请在 `data/` 下检查 `Atlas_*.nii` 的标签范围（示例数据集中标签 ~170）。

训练与实验流程
----------------
典型实验分为：
1. Patch 级预训练（学习 patch 表征）。
2. 迭代重标（relabel）：用高置信预测更新 patch 伪标签并微调。用于清洗 noisy labels。  
3. 微调阶段（冻结骨干 -> 解冻联合训练）。
4. Stage5（可选）：训练独立 `PatchMILAggregator`（attention），生成 `aggregator_best.pth`，用于提升患者级性能。

常用命令
-----------

训练
```bash
PYTHONUNBUFFERED=1 python -u train.py --config config.yaml --out_dir outputs/exp4
```

评估（患者级）
```bash
python eval_patient_level.py --exp_dir outputs/exp4
# 如果存在 outputs/exp4/aggregator_best.pth，脚本会自动加载并使用 attention 聚合
```

单病人推理与解释
```bash
python inference.py --ckpt outputs/exp4/final_best.pth --qsm path/qsm.nii --t1 path/t1.nii --aal path/aal.nii --out_dir outputs/infer/PD001
```

关键输出说明
----------------
- `outputs/<exp>/final_best.pth` : Patch 分类器权重（ResNet3D）。
- `outputs/<exp>/aggregator_best.pth` : stage5 训练的 attention 聚合器权重（如果训练过）。
- `outputs/<exp>/explain_out/<pid>/voxel_importances.npz` : voxel 级重要性热图（可转为 NIfTI）。
- `outputs/<exp>/explain_out/<pid>/top_nodes.json` : top-k 重要节点（ROI/sv/slice）。
- `outputs/<exp>/eval_patient_level.txt` : 患者级评估结果（AUC/ACC/SEN/PRE/F1/SPE）。

聚合策略比对
----------------
- `topk_mean`：默认、稳健，适合多数场景。  
- `attention`：需训练 `aggregator_best.pth`（stage5），可显著提高召回与总体 AUC（见实验 `exp4-attention`）。  
- `overlap-aware`：考虑重叠 patch 的 voxel 投票到 ROI 的方案，可能改善局部定位但需谨慎调参以避免假阳性。

解释器工作流（简述）
----------------------
1. 对指定层（默认 `layer4`）做 Grad-CAM，得到 feature-map 热图。  
2. 使用 supervoxel（SLIC）或 AAL mask 将热图映射到图节点（ROI/slice/sv）。
3. 训练 Tiny GNN（多次重复平均）得到节点/边重要性分数。  
4. 输出：`voxel_importances.npz`, `top_nodes.json`, `node_importances.csv`, `top_nodes.png`。

如何快速检查解释结果
----------------------
- 把 `voxel_importances.npz` 转为 NIfTI：
  ```python
  import numpy as np, nibabel as nib
  imp = np.load('voxel_importances.npz')['arr_0']  # key 可能不同
  nib.save(nib.Nifti1Image(imp.astype('float32'), np.eye(4)), 'voxel_importances.nii.gz')
  ```
- 在 ITK-SNAP/3D Slicer 中将其叠加到 QSM/T1 做检查；用 `top_nodes.png` 做快速人工复核。

提升模型性能的可执行建议（由易到难）
---------------------------------
1. 阈值优化（最快）：对每种聚合在验证集上做 Youden / F1-opt 阈值搜索。  
2. 概率校准：Platt scaling / temperature scaling。  
3. 后处理：小簇过滤（剔除小体积连通块）、ROI 权重调整。  
4. 聚合混合：在验证集上用线性模型学习 topk_mean 与 attention 的融合权重。  
5. 损失与采样：focal loss、class-weight、难样本挖掘。  
6. 集成（长期）：k-fold ensemble 或 TTA。

调试与常见问题
-----------------
- 日志文件无输出：使用 `PYTHONUNBUFFERED=1` 或 `python -u` 启动，或 `nohup python -u ... > out.log 2>&1 &`。  
- AAL/Atlas 标签不匹配：检查 `data/*/Atlas_*.nii` 的唯一标签范围，若有多个 atlas 版本请在 manifest 中明确指定路径。  
- 进程占用但日志为空：可能是 stdout 被缓冲，使用 `-u` 或 `stdbuf -oL` 强制刷新。

进一步阅读与扩展
-------------------
- 若希望把 README 翻译为英文版或生成一页快速操作手册（带常用命令与路径），请告诉我，我会补充。

联系方式与许可
-----------------
该仓库用于研究目的；使用医学数据时请遵守伦理/隐私规定。如需合作或内部支持，请联系仓库维护者（OWNER）。

"""
```bash
