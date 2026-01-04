#!/usr/bin/env python3
"""
Plot MULTIPLE ROC curves in one figure with customizable styles.
Features:
  - Accepts multiple experiment directories.
  - Automatically assigns different colors to each curve.
  - Allows specifying one curve to be bold.
  - Draws step-like ROC curves (common in ML papers).
  - Customizable title, labels, and legend style.

Usage:
  python scripts/plot_roc_distribution.py \
       --exp_dir outputs/exp_K1 outputs/exp_K2 outputs/exp_K3 outputs/exp_K4 outputs/exp_K5 \
       --labels "K=1" "K=2" "K=3" "K=4" "K=5" \
       --bold_label "K=5" \
       --title "ROC Curves for Different K-values" \
       --device cuda \
       --out outputs/roc_multiple_final.png
"""
import os
import argparse
import yaml
import torch
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import bootstrap

# Assuming these modules exist in your project
from data_utils import PDVolDataset, BagDataset
from models_patch import build_patch_model, PatchMILAggregator


def find_ckpt(exp_dir, patterns):
    for p in patterns:
        path = os.path.join(exp_dir, p)
        if os.path.exists(path):
            return path
    for fname in os.listdir(exp_dir):
        for p in patterns:
            if p in fname:
                return os.path.join(exp_dir, fname)
    return None


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def process_single_exp(exp_dir, device, batch_size):
    """Process a single experiment directory and return truths and probabilities."""
    exp_cfg_path = os.path.join(exp_dir, 'config.yaml')
    if not os.path.exists(exp_cfg_path):
        raise FileNotFoundError(f"config.yaml not found in experiment directory {exp_dir}")
    cfg = load_yaml(exp_cfg_path)

    backbone = build_patch_model(cfg).to(device)
    backbone.eval()
    feat_dim = backbone.fc.in_features

    train_cfg = cfg.get('train', {})
    aggregator = PatchMILAggregator(
        feat_dim=feat_dim, pooling='distribution',
        topk_percent=train_cfg.get('topk_percent', 0.15),
        tau=train_cfg.get('tau', 0.1), q=train_cfg.get('q', 0.9),
        q_min=train_cfg.get('q_min', 0.5), q_ref_n=train_cfg.get('q_ref_n', 100)
    ).to(device)
    aggregator.eval()

    # Load Backbone
    bb_ckpt = find_ckpt(exp_dir, ['relabel_iter1.pth', 'pretrain_best.pth', 'final_best.pth', 'relabel_iter2.pth'])
    if not bb_ckpt: raise FileNotFoundError(f"Backbone checkpoint not found in {exp_dir}")
    bb_state = torch.load(bb_ckpt, map_location='cpu').get('state_dict', torch.load(bb_ckpt, map_location='cpu'))
    try: backbone.load_state_dict(bb_state)
    except: backbone.load_state_dict({k.replace('module.', ''): v for k, v in bb_state.items()})

    # Load Aggregator
    agg_ckpt = find_ckpt(exp_dir, ['aggregator_distribution_best.pth', 'aggregator_best.pth'])
    if not agg_ckpt: raise FileNotFoundError(f"Aggregator checkpoint not found in {exp_dir}")
    agg_state = torch.load(agg_ckpt, map_location='cpu').get('state_dict', torch.load(agg_ckpt, map_location='cpu'))
    try: aggregator.load_state_dict(agg_state)
    except: aggregator.load_state_dict({k.replace('module.', ''): v for k, v in agg_state.items()})

    # Load Dataset
    val_csv = cfg['data'].get('val_csv')
    if not val_csv: raise ValueError(f"val_csv not set in config for {exp_dir}")
    val_ds = PDVolDataset(manifest_csv=val_csv, input_size=tuple(cfg['data'].get('input_size', [192,192,128])), mode='eval')
    bag_ds = BagDataset(val_ds, patch_size=tuple(train_cfg.get('patch_size', [48,48,32])), stride=tuple(train_cfg.get('patch_stride', train_cfg.get('patch_size', [48,48,32]))))

    truths, probs = [], []
    for i in tqdm(range(len(bag_ds)), desc=f'Processing {os.path.basename(exp_dir)}'):
        sample = bag_ds[i]
        patches, label = sample['patches'], int(sample['label'])
        
        feats, logits = [], []
        with torch.no_grad():
            for start in range(0, patches.shape[0], batch_size):
                end = min(patches.shape[0], start + batch_size)
                batch_logit, batch_feat = backbone(patches[start:end].to(device), return_features=True)
                feats.append(batch_feat.cpu()), logits.append(batch_logit.cpu())
        
        feats, logits = torch.cat(feats).unsqueeze(0).to(device), torch.cat(logits).unsqueeze(0).to(device)
        with torch.no_grad():
            patient_prob = torch.sigmoid(aggregator(logits, feats)).item()
        
        truths.append(label), probs.append(patient_prob)

    csv_path = os.path.join(exp_dir, 'patient_probs.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows([['label', 'prob']] + list(zip(truths, probs)))
    print(f"[INFO] Probabilities for {exp_dir} saved to {csv_path}")

    return truths, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True, nargs='+', help='List of experiment directories.')
    ap.add_argument('--labels', nargs='+', help='Labels for each experiment (must match the number of exp_dirs). Defaults to directory names.')
    ap.add_argument('--bold_label', default=None, help='Label of the curve to be bolded (must match one of the --labels).')
    ap.add_argument('--title', default='ROC Curves Comparison', help='Title of the plot.')
    ap.add_argument('--device', default='cuda', help='Device to use for computation (cuda/cpu).')
    ap.add_argument('--batch_size', type=int, default=64, help='Batch size for patch feature extraction.')
    ap.add_argument('--out', default='combined_roc.png', help='Output path for the ROC plot.')
    args = ap.parse_args()

    # Validate and set labels
    if not args.labels:
        args.labels = [os.path.basename(exp) for exp in args.exp_dir]
    if len(args.labels) != len(args.exp_dir):
        raise ValueError(f"Number of labels ({len(args.labels)}) does not match number of experiments ({len(args.exp_dir)}).")

    # Process all experiments
    roc_list = []
    for exp_dir, label in zip(args.exp_dir, args.labels):
        truths, probs = process_single_exp(exp_dir, args.device, args.batch_size)
        truths = np.array(truths)
        probs = np.array(probs)
        fpr, tpr, _ = roc_curve(truths, probs)
        roc_auc = auc(fpr, tpr)
        
        # Compute 95% CI for AUC using bootstrap
        def auc_func(data):
            y_true, y_score = data
            fpr_b, tpr_b, _ = roc_curve(y_true, y_score)
            return auc(fpr_b, tpr_b)
        data = (truths, probs)
        try:
            res = bootstrap((data,), auc_func, n_resamples=1000, confidence_level=0.95, method='percentile')
            ci_low, ci_high = res.confidence_interval
            auc_str = f"{roc_auc:.3f} (95% CI: {ci_low:.3f}–{ci_high:.3f})"
        except:
            auc_str = f"{roc_auc:.3f}"
        
        roc_list.append((fpr, tpr, roc_auc, auc_str, label))

    # --- Plotting ---
    plt.style.use('default') # Reset style
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)  # Square figure

    # Define color palette: low saturation, colorblind-friendly
    colors = plt.cm.get_cmap('Set1', len(roc_list))
    
    # Plot each ROC curve
    for i, (fpr, tpr, roc_auc, auc_str, label) in enumerate(roc_list):
        linewidth = 2.5 if (args.bold_label and label == args.bold_label) else 2.0
        linestyle = '-' if i == 0 else '--'  # Main model solid, others dashed
        ax.plot(fpr, tpr, 
                color=colors(i), 
                lw=linewidth, 
                linestyle=linestyle,
                label=f'{label} (AUC = {auc_str})',
                drawstyle='steps-post')  # Step-like for medical style

    # Plot the chance baseline
    ax.plot([0, 1], [0, 1], 
            linestyle=':', 
            color='gray', 
            lw=1.5, 
            label='Chance (AUC = 0.500)')

    # Set plot attributes
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(args.title, fontsize=14, fontweight='bold')
    
    # Legend: lower right, no frame
    ax.legend(loc='lower right', fontsize=10, frameon=False)
    
    # No grid for cleaner look
    # ax.grid(True, linestyle='--', alpha=0.6)

    # Ensure square aspect
    ax.set_aspect('equal', adjustable='box')
    
    # Save with high quality
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches='tight', pad_inches=0.1, dpi=300)
    print(f"\n[SUCCESS] Combined ROC plot saved to: {args.out}")


if __name__ == '__main__':
    main()