# 3D model + Post-hoc Graph Explainer for PD diagnosis

Quick start:
1. Install dependencies from requirements.txt
2. Prepare dataset manifests (3-channel inputs):
   - data/train_manifest.csv and data/val_manifest.csv with rows: qsm_path,t1_path,aal_path,label,id
   - The model expects 3 channels per sample: QSM, T1w and an ROI/AAL segmentation map as the third channel.
3. Edit config.yaml to point to your files and hyperparams.
4. Train (small-sample friendly):
   - `config.yaml` contains small-sample options: `model.norm` (group/batch), `model.dropout`, `train.balanced_sampler`, `train.augmentations`.
   - Example run (synthetic smoke-test):
   ```bash
   python train.py --synthetic --max-epochs 10 --device cuda
   ```
   Best checkpoint saved to ./checkpoints/best_checkpoint.pth
5. Run inference + explanation:
   ```
   python inference.py
   ```
    Edit arguments inside inference.py or call run_inference_and_explain(...) from interactive session.

## VS Code quick workflow (one-click runs)

If you're using VS Code (recommended with Remote‑SSH for server work), this repo includes a `.vscode/tasks.json` and `.vscode/launch.json` so you can run common actions with one click or debug.

- `Tasks` available (Run → Command Palette → Tasks: Run Task):
   - Train (GPU 0) — runs `train.py` with `config.yaml` and saves to `outputs/exp1`
   - Inference (single patient) — runs `inference.py` and writes to `outputs/infer/<patientId>` (prompts for paths)
   - Run localizer — converts voxel_importances into masks and visualizations
   - TensorBoard — opens tensorboard for `outputs/exp1/logs`
   - Push to origin — helper task that runs `git add && git commit && git push`

- `Launch` configurations (Run/Debug panel):
   - Python: Debug Training — debug `train.py` using `config.yaml`
   - Python: Debug Inference — debug `inference.py` against a sample patient

## Push your local code to GitHub (git@github.com:gen124/mytest.git)

If you want to push this repository to `git@github.com:gen124/mytest.git` use the steps below. IMPORTANT: do not push raw patient data or anything with PHI — ensure `data/` and `outputs/` are listed in `.gitignore`.

1) Check `.gitignore` exists and contains data/ and outputs/ entries (this repo already has a safe `.gitignore`).

2) Commit & add remote (run locally):

```bash
git add .
git commit -m "Initial project sync"
git remote add origin git@github.com:gen124/mytest.git
git push -u origin main
```

If `main` doesn't exist on remote, use your branch name, or create `main` locally before pushing.

3) On the server: clone the repository and open with Remote-SSH / run directly.

```bash
cd /home/youruser/projects
git clone git@github.com:gen124/mytest.git
cd mytest
# Create conda env and install deps
conda create -n 3dgraph python=3.10 -y
conda activate 3dgraph
pip install -r requirements.txt
```

See earlier sections of this README for data manifest format (qsm,t1,aal,label,id), recommended paths, and precautions about PHI.
