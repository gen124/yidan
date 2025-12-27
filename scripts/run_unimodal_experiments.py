#!/usr/bin/env python3
"""
自动生成 T1-only / QSM-only manifest，并在 GPU0 上依次训练和评估，输出最终表格。
- manifest 生成规则：T1-only 时 QSM 路径全零，QSM-only 时 T1 路径全零（或空字符串，data_utils 会自动补零）。
- 训练/评估命令自动调用 train.py 和 eval_patient_level.py。
- 评估结果自动汇总为表格。
"""
import os
import csv
import argparse
import shutil
import yaml
import nibabel as nib
import numpy as np
import subprocess

def read_manifest(path):
    rows = []
    with open(path, 'r') as f:
        rdr = csv.reader(f)
        for r in rdr:
            if not r:
                continue
            rows.append(r)
    return rows

def write_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)

def make_zero_nifti(out_path, shape):
    H, W, D = shape[0], shape[1], shape[2]
    arr = np.zeros((D, H, W), dtype=np.float32)
    nii = nib.Nifti1Image(arr, affine=np.eye(4))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nii, out_path)

def prepare_unimodal_manifests(orig_manifest, zero_qsm_path, mode):
    rows = read_manifest(orig_manifest)
    out_rows = []
    for r in rows:
        qsm, t1, aal, label, pid = r[0], r[1], r[2], r[3], r[4]
        if mode == 't1':
            qsm_new = zero_qsm_path
            t1_new = t1
        else:
            qsm_new = qsm
            t1_new = ''
        out_rows.append([qsm_new, t1_new, aal, label, pid])
    return out_rows

def copy_and_patch_config(orig_cfg_path, out_cfg_path, train_csv, val_csv):
    with open(orig_cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['data']['train_csv'] = train_csv
    cfg['data']['val_csv'] = val_csv
    os.makedirs(os.path.dirname(out_cfg_path), exist_ok=True)
    with open(out_cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)

def run_cmd(cmd, env=None):
    print("RUN:", " ".join(cmd))
    p = subprocess.Popen(cmd, env=env)
    p.wait()
    return p.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--out_dir', default='outputs/unimodal')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    base_out = args.out_dir
    os.makedirs(base_out, exist_ok=True)
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    train_manifest = cfg['data']['train_csv']
    val_manifest = cfg['data']['val_csv']
    input_size = cfg['data'].get('input_size', [192,192,128])
    zero_path = os.path.join('data', 'zero_qsm.nii.gz')
    if not os.path.exists(zero_path):
        print(f'Create zero nifti at {zero_path} with input_size {input_size}')
        make_zero_nifti(zero_path, input_size)
    else:
        print(f'Zero nifti already exists: {zero_path}')
    results = []
    for mode in ['t1', 'qsm']:
        mode_out = os.path.join(base_out, mode)
        manifests_out = os.path.join(mode_out, 'manifests')
        os.makedirs(manifests_out, exist_ok=True)
        train_rows = prepare_unimodal_manifests(train_manifest, zero_path, mode)
        val_rows = prepare_unimodal_manifests(val_manifest, zero_path, mode)
        write_manifest(os.path.join(manifests_out, 'train_manifest.csv'), train_rows)
        write_manifest(os.path.join(manifests_out, 'val_manifest.csv'), val_rows)
        cfg_out_path = os.path.join(mode_out, 'config.yaml')
        copy_and_patch_config(args.config, cfg_out_path, os.path.join(manifests_out,'train_manifest.csv'), os.path.join(manifests_out,'val_manifest.csv'))
        print(f'Prepared {mode} experiment in {mode_out}')
        # Do NOT run training/eval automatically here. Only generate manifests and patched config.
        # Print suggested commands for manual background execution (nohup/tmux) using GPU index provided.
        print('\nPrepared experiment (no training started). Next steps:')
        print('  # Train (background, set GPU index as needed)')
        print(f"  CUDA_VISIBLE_DEVICES={args.gpu} nohup python train.py --config {cfg_out_path} --out_dir {mode_out} > {mode_out}/train.log 2>&1 & echo $!")
        print('  # After training finishes, evaluate (background)')
        print(f"  CUDA_VISIBLE_DEVICES={args.gpu} nohup python eval_patient_level.py --exp_dir {mode_out} --config {cfg_out_path} > {mode_out}/eval.log 2>&1 & echo $!\n")
    print('\nDone. Check outputs/unimodal/t1 and outputs/unimodal/qsm for results.')

if __name__ == '__main__':
    main()
