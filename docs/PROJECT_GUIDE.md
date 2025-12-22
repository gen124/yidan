# PatchMIL-3D 项目全景指南

## 1. 项目综述与新意
- 核心目标：在小样本、多模态（QSM+T1+Atlas）场景下，通过 Patch 级伪标签预训练 + 迭代重标 + MIL 聚合，实现患者级 PD 诊断。
- 设计动机与效果
  - Patch 级伪标签预训练：把患者标签下放到 patch，等价于“数据放大器”，先学到稳定局部表征，减轻直接患者级训练的过拟合。
  - 迭代重标：高置信收正、低置信收负，逐轮修正伪标签，降低噪声；实测在 2-3 轮后 AUC 持续抬升。
  - 分阶段微调：先冻结骨干只训头部，再解冻小 lr 联合训练，能保护已学到的通用特征，避免小样本灾难性遗忘。
  - 重叠 Patch MIL：当 stride < size 时，引入体素级加权 + ROI 聚合，再做患者决策，能缓解 patch 边界割裂，提升定位一致性。
  - 缺模态鲁棒：T1/Atlas 缺失时零填，避免样本被丢弃，保证召回；QSM 必需保持对比度信息。
  - 自动 batch/loader 调参：一次试跑估计显存/CPU，并行度自适应，提高吞吐；省去手调尝试。
  - 小样本友好：多级正则（dropout、分阶段训练）+ 重标降噪，使在 50-100 例规模仍可稳定训练。

## 2. 目录结构（工作区）
- 核心代码
  - [train.py](../train.py) 4 阶段训练（预训练→迭代重标→阶段微调）。
  - [data_utils.py](../data_utils.py) 数据加载、PatchDataset、缺模态处理。
  - [models_patch.py](../models_patch.py) ResNet3D Patch 分类器 + `PatchMILAggregator`（max/topk_mean/attention）。
  - [mil_overlap_utils.py](../mil_overlap_utils.py) 重叠/非重叠 MIL 推理管道。
  - [inference.py](../inference.py) / [inference_overlap_example.py](../inference_overlap_example.py) 推理示例。
  - [metrics.py](../metrics.py) AUC/ACC 评估。
- 配置与依赖
  - [config.yaml](../config.yaml) 主配置（数据路径、阈值、loader、模型、patch 设置）。
  - [requirements.txt](../requirements.txt) 依赖列表。
- 数据与输出
  - `data/` 已解压的 HC/PD 目录（QSM/T1/Atlas）；`data/train_manifest.csv`, `data/val_manifest.csv`。
  - `outputs/` 训练日志与检查点；`checkpoints/` 快速权重存放。

## 3. 数据准备（为何这样组织）
- 目录组织：
  - `data/HC/HCxxx/` 内含 `QSM_*.nii`, `T1_*.nii`, `Atlas_*.nii`（T1/Atlas 可缺失）。
  - `data/PD/PDxxx/` 同上。
- 清单文件（已生成）
  - 训练清单：[data/train_manifest.csv](../data/train_manifest.csv)
  - 验证清单：[data/val_manifest.csv](../data/val_manifest.csv)
  - 每行：`qsm_path,t1_path,aal_path,label,id`；`label`：PD=1，HC=0；缺失模态留空字符串。
- 这样设计的原因：
  - 显式清单文件方便可复现的切分与追踪；缺失模态用空字符串使加载端能零填且不中断。
  - 保持 QSM 路径必须存在，保证最关键模态质量；其余模态缺失不致使样本流失。
- 若需重建清单（8:2 分层）：
  ```bash
  /home/sunyidan/miniconda3/envs/patchmil-gpu/bin/python - <<'PY'
  import os, csv, random
  root = '/home/sunyidan/PatchMIL-3D/data'
  label_map = {'HC': 0, 'PD': 1}
  entries = {0: [], 1: []}
  for grp, lab in label_map.items():
      gdir = os.path.join(root, grp)
      for subj in sorted(os.listdir(gdir)):
          sdir = os.path.join(gdir, subj)
          qsm = os.path.join(sdir, f'QSM_{subj}.nii')
          t1  = os.path.join(sdir, f'T1_{subj}.nii')
          atl = os.path.join(sdir, f'Atlas_{subj}.nii')
          if not os.path.exists(qsm):
              continue
          entries[lab].append((qsm, t1 if os.path.exists(t1) else '', atl if os.path.exists(atl) else '', lab, subj))
  random.seed(42)
  train, val = [], []
  for lab, rows in entries.items():
      random.shuffle(rows)
      n = len(rows); n_val = max(1, round(n * 0.2))
      val += rows[:n_val]; train += rows[n_val:]
  for path, rows in [('/home/sunyidan/PatchMIL-3D/data/train_manifest.csv', train),
                     ('/home/sunyidan/PatchMIL-3D/data/val_manifest.csv', val)]:
      with open(path, 'w', newline='') as f: csv.writer(f).writerows(rows)
  PY
  ```

## 4. 配置要点（当前生效值与理由）
- 数据：
  - `data.train_csv`: `./data/train_manifest.csv`
  - `data.val_csv`: `./data/val_manifest.csv`
  - `data.input_size`: `[192,192,128]`（中心裁剪/填充）
- 训练：
  - `batch_size=24`（auto-tune 推到 96）；`num_workers=16`，`prefetch_factor=6`，`pin_memory=true`，`persistent_workers=true`；`amp=true`。
  - 迭代重标：`n_relabel_iters=3`；`high_conf_threshold=0.9`，`low_conf_threshold=0.3`（收紧正以减少伪正，放宽负以捕获更多负样本）。
  - 阶段：`pretrain_epochs=20`，`finetune_epochs=5/iter`，`head_finetune_epochs=5`，`final_epochs=10`。
  - Patch：`patch_size=[48,48,32]`，`patch_stride=[48,48,32]`（可改成重叠以启用 overlap MIL）。
- 模型：ResNet3D (base_channels=16, GroupNorm, dropout=0.3)。
- 原因与效果：
  - 大 batch（通过 auto-tune 达 96）+ AMP：提升吞吐并平滑梯度；auto-tune 以 80% 显存为目标，减少 OOM 调参成本。
  - 多 worker + 高 prefetch：缓解 IO 瓶颈，避免 GPU 等数据；persistent_workers 降低 epoch 间重建开销。
  - 高/低阈值分离：提高正阈值抑制伪正，放宽负阈值便于早期收集负样本，避免“全正”重标。
  - GroupNorm：小 batch 稳定；dropout 控制过拟合。
  - 重叠 stride（可选）+ overlap MIL：用于需要更细粒度定位时，代价是更多算力。

## 5. 训练与验证（流程与背后逻辑）
- 启动（建议无缓冲日志）：
  ```bash
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
  python -u train.py --config config.yaml --out_dir outputs/patch_mil
  ```
- 训练流程（4 阶段）
  1) Patch 伪标签预训练：先学“粗”表征，降低后续重标的冷启动风险。
  2) 迭代重标：用模型自身置信度清洗伪标签，迭代提升标签质量，再微调；相当于自训练的稳健版。
  3) 阶段微调-1：冻结骨干，只训头部，保护已学底层特征。
  4) 阶段微调-2：解冻骨干，小 lr 联合训练，微调全局。
- 输出
  - `pretrain_best.pth`, `relabel_iter{k}.pth`, `final_best.pth`
  - `metadata.json`: 训练元信息
  - 日志：`outputs/patch_mil/train_gpu1.log`（根据启动命令）
- 监控
  - `tail -f outputs/patch_mil/train_gpu1.log`
  - `watch -n 5 nvidia-smi` 或 `nvidia-smi dmon -i 1 -s pucmt -d 2`

## 6. 推理与 MIL 聚合（可解释性与选择）
- 基本推理（单患者）：
  ```bash
  python inference.py \
    --ckpt outputs/patch_mil/final_best.pth \
    --qsm /path/to/QSM.nii \
    --t1 /path/to/T1.nii \
    --aal /path/to/Atlas.nii \
    --out_dir outputs/infer/p001
  ```
- 重叠/非重叠自适应 MIL：`complete_overlap_mil_pipeline` ([mil_overlap_utils.py](../mil_overlap_utils.py))
  - `pooling` 选项：`max` / `topk_mean`（默认 0.15）/ `attention`（需提供特征）。
  - 重叠启用：将 `patch_stride` 设小于 `patch_size`。
- 示例脚本：见 [inference_overlap_example.py](../inference_overlap_example.py)。
- 为什么这样：
  - topk_mean：对噪声鲁棒，兼顾多处病灶；k 比例可调平衡“聚焦”与“覆盖”。
  - max：符合 MIL 经典假设，最敏感但易受单点噪声影响。
  - attention：可学权重，若提供 patch 特征，可自适应聚合；需在管道中传特征。
  - 重叠模式：先体素级融合再 ROI 聚合，可提升空间一致性与可解释性（热图更平滑）。

## 7. 性能与验证
- 指标：AUC、ACC（见日志和 `metadata.json`）。
- 验证集由 manifest 指定；可通过调整阈值或增加重标轮次提升 AUC。

## 8. 性能加速建议（动作与预期效果）
- DataLoader：已调至 `num_workers=16, prefetch_factor=6`；若仍慢，可尝试 `prefetch_factor=8` 或减小 `batch_size` 防止抖动。
- IO：数据放 NVMe；如需进一步加速，可离线缓存 NIfTI→`.npy`。
- 混合精度：保持 `amp=true`。
- 预期效果：
  - 提高 prefetch/worker：减少 GPU 等数据；但过高会争抢 CPU/内存，需要监控负载。
  - 缓存为 `.npy`：典型能将加载开销降至原来的 30-50%。
  - AMP：显存与算速双收益，避免 OOM 风险时可下调 batch。

## 9. 常见问题
- 无负样本被重标：提高 `high_conf_threshold`（已 0.9），放宽 `low_conf_threshold`（已 0.3）；或再训练若干 epoch 再重标。
- 训练过慢：提高 `num_workers/prefetch`，检查磁盘 IO，或降低 batch_size。
- 显存不足：降低 auto-tune 目标（可在代码里改 80%→70%），或减小 `batch_size/patch_size`。

## 10. 下一步可改进
- 支持 `--resume`（保存/加载优化器与 epoch）以无损改参重启。
- 在重标前重读阈值配置，实现阶段内热更新。
- 推理时暴露 CLI 切换 `pooling` 和 `topk_percent`。
