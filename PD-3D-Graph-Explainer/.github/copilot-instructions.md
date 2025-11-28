## Quick context — what this project does

A 3D ResNet encoder trains on 3-channel medical volumes (QSM + T1 + ROI segmentation) and outputs a patient-level score. See `models.py` (ResNet3D_withSliceHead) and `data_utils.py`.
- Post-hoc explanation pipeline converts the ResNet feature-map into small graphs (ROI / slice / supervoxels) and uses a tiny GNN explainer to compute node/edge importances. See `graph_builder.py`, `explainer.py` and `inference.py`.

## High-level architecture & dataflow (important files)

- data ingestion & transforms: `data_utils.py`
  - Expected manifest format: `qsm_path,t1_path,aal_path,label,id` (README.md).
  - Input volumes normalized with `zscore_normalize` and center-cropped/padded to shape from `config.yaml`.
Encoder & heads: `models.py`
  - `ResNet3D_withSliceHead` is now a fully 3D model (no 2D slice head). Use `return_feat_map=True` to retrieve spatial feature maps for explainability.
  - The model supports small-batch-friendly normalization (GroupNorm) and an optional dropout — controlled by `config.yaml` (`model.norm`, `model.num_groups`, `model.dropout`).
 - data ingestion & transforms: `data_utils.py`
  - Input volumes normalized with `zscore_normalize` and center-cropped/padded to shape from `config.yaml`.
  - Dataset supports light augmentations useful for small-sample training (flip, small gaussian noise). Pass these with `train.augmentations` in `config.yaml`.
 - Explainer & helper: `explainer.py`
  - `fit_tiny_gnn_and_explain(nodes, feats, edges, model_prob, device, n_runs, noise_std)` — trains a tiny GAT-based GNN and returns node and edge importances. Use `n_runs>1` and tiny `noise_std` to get more stable explanations on small datasets.
  - `path_pdm(node_scores, edges, topk)` — produces likely high-importance node paths from node importances and edge weights.
- Orchestration: `inference.py`
  - Loads checkpoint, runs Grad-CAM (`grad_cam_3d`) against `layer4`, builds graphs from `feat_map`, runs tiny GNN explainer and saves results to `explain_out`.

## Key developer workflows & commands

- Install dependencies:

  pip install -r requirements.txt

- Train (README refers to `train.py`) — typical flow:
  - Prepare manifests: `data/train_manifest.csv`, `data/val_manifest.csv` rows must be `qsm_path,t1_path,aal_path,label,id`.
  - Edit `config.yaml` (models / train / explainer sections) before runs.
  - Run training script (project expects `train.py`) or run from your preferred trainer loop.

- Run inference & explain (direct API example):
  - Call from python / REPL: run_inference_and_explain(model_ckpt, qsm_path, t1_path, aal_path, cfg_path='config.yaml')
  - Or run `python inference.py` after editing the script-level arguments.

## Project-specific conventions & patterns agents should follow

- Feature & shape expectations:
  - Input volume shape (before batch): (C=3, D, H, W) — code stacks QSM, T1 and ROI segmentation channels.
  - Model forward passes often return tuples — check callsites (e.g., `model(x)` may return logits or tuple). Use `if isinstance(out, tuple)` guards.
  - Feature-map `feat_map` has shape (B, C, Df, Hf, Wf). `inference.py` averages spatial dims to build slice logits.

- Graphs representation (used by explainer):
  - Nodes: list of strings 'roi_#', 'slice_#', or 'sv_#'
  - Node features: numpy array shape (N, C)
  - Edges: list of (u, v, weight) where weight is float similarity/adjacency
  - Explainer expects a scalar model probability `model_prob` when training the small GNN.

- Checkpoint loading: `inference.py` handles both raw state dict and wrapped checkpoints (`ck['state_dict']`). Mirror this behavior when loading elsewhere.

## Debugging tips / common gotchas

- Device fallback: `inference.py` picks GPU if available (via `cfg['train']['device']`) else CPU. Tests on CPU may require lowering batch sizes and epochs.
- Grad-CAM hook: `register_forward_hook` uses `layer4` by default and expects the module to exist; hook fallback is the last module in `named_modules()`.
- Supervoxel graphs use `skimage.segmentation.slic` — ensure `scikit-image` in `requirements.txt` and input data / `compactness` param produce expected segment counts.

## Where to look next (quick pointers for new contributors)

- To extend explainer behavior, modify `explainer.py` (TinyGNN architecture and training schedule) and `inference.py` (how `feat_map` is converted into graphs).
- To add dataset variants or new augmentation, update `data_utils.PDVolDataset` and `config.yaml` with new hyperparameters.

If anything here seems wrong or incomplete, tell me which area you'd prefer expanded (e.g., add code snippets for common test runs or CI steps).
