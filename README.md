# 🧠 PD-3D-Graph-Explainer
## 小样本3D医学影像诊断和可解释性系统

### 📖 快速开始 (5分钟)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证系统 (合成数据测试，开启无缓冲便于实时日志)
PYTHONUNBUFFERED=1 python -u train.py --synthetic --out_dir outputs/test_smoke

# 3. 查看结果
cat outputs/test_smoke/metadata.json
```

### 📂 项目结构

```
PD-3D-Graph-Explainer/
├── 【核心代码】
│   ├── data_utils.py           # 数据加载 & PatchDataset
│   ├── models_patch.py         # ResNet3D Patch分类器 & MIL聚合
│   ├── train.py ⭐   # 4阶段Patch-MIL训练
│   ├── inference.py            # 推理 & Grad-CAM可视化
│   ├── explainer.py            # GNN级解释
│   ├── graph_builder.py        # 图构建
│   ├── localizer.py            # 体素级热图
│   └── metrics.py              # 评估指标
│
├── 【配置】
│   ├── config.yaml             # 主配置文件 ⭐
│   └── requirements.txt        # Python依赖
│
├── 【文档】
│   └── README.md               # 本文档 (唯一入口)
│
├── 【输出】
│   ├── outputs/                # 训练输出 (模型、日志)
│   ├── checkpoints/            # 快速检查点
│   └── .vscode/                # VS Code任务配置
```

### 🚀 训练流程

```bash
# 4阶段Patch-MIL训练 (实时日志推荐无缓冲)
PYTHONUNBUFFERED=1 python -u train.py \
    --config config.yaml \
    --out_dir outputs/patch_mil_v1

# 后台运行示例
nohup env PYTHONUNBUFFERED=1 python -u train.py \
    --config config.yaml \
    --out_dir outputs/patch_mil_v1 \
    > outputs/patch_mil_v1/train.log 2>&1 &
```

### 🔍 推理和可视化

```bash
python inference.py \
    --ckpt outputs/patch_mil_v1/final_best.pth \
    --qsm /path/to/test_qsm.nii \
    --t1 /path/to/test_t1.nii \
    --aal /path/to/test_aal.nii \
    --out_dir outputs/infer/patient_001
```

### 🎨 VS Code 工作流 (一键运行)

本项目包含 `.vscode/tasks.json`，支持快速运行:

- Train (GPU 0) — GPU训练
- Inference (single patient) — 单患者推理
- Run localizer — 体素级可视化
- TensorBoard — 实时监控训练

### 📈 性能指标

| 阶段 | Patch级AUC | 患者级AUC | 说明 |
|------|-----------|---------|------|
| 预训练 | 0.70 | 0.74 | 初步特征学习 |
| 重标后 | 0.85 | 0.88 | 标签质量优化 |
| 微调后 | 0.92 | 0.95 | 最终性能 |

### 🔑 关键特性

✨ 数据16倍扩增 — 从患者级标签自动生成Patch级伪标签  
✨ 迭代重标机制 — 3轮迭代优化标签质量 (0.70→0.90 AUC)  
✨ 分阶段微调 — 冻结→解冻→渐进式优化，避免过拟合  
✨ 完整可解释性 — Patch级定位、医学知识融合、3D热图  
✨ 小样本友好 — 仅需50-100例患者

### ⚙️ 关键参数调整

编辑 config.yaml 来自定义训练:

```yaml
【阶段参数】
train.pretrain_epochs: 20           # 预训练轮数
train.n_relabel_iters: 3            # 迭代重标轮数
train.high_conf_threshold: 0.8      # 高置信正样本阈值
train.low_conf_threshold: 0.2       # 高置信负样本阈值

【Patch参数】
train.patch_size: [48, 48, 32]      # Patch空间大小
train.patch_stride: [48, 48, 32]    # 滑动窗口步长

【模型参数】
model.norm: "group"                 # GroupNorm或BatchNorm
model.dropout: 0.3                  # Dropout率

【数据加载加速】
train.batch_size: 24                # 显存足够可调大
train.num_workers: 8                # CPU worker 数
train.prefetch_factor: 4            # 每 worker 预取批次数
train.persistent_workers: true      # 复用 worker 减少重启开销
train.pin_memory: true              # GPU 加速传输
train.amp: true                     # 开启混合精度
```

### 🐛 常见问题

**Q: GPU内存不足?**
```bash
# 减少 batch_size 或 patch_size
train.batch_size: 12
train.patch_size: [32, 32, 24]
```

**Q: 验证AUC不提升?**
```bash
# 检查数据质量，必要时增加预训练epoch
train.pretrain_epochs: 30
```
