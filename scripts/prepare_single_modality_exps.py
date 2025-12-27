#!/usr/bin/env python3
"""生成单模态训练用的 manifest 与 config 副本：T1-only 与 QSM-only

用法示例：
  python scripts/prepare_single_modality_exps.py --base_config config.yaml --out_dir outputs/exp4_singlemod

生成内容：
  outputs/exp4_singlemod/manifests/train_T1.csv
  outputs/exp4_singlemod/manifests/val_T1.csv
  outputs/exp4_singlemod/manifests/train_QSM.csv
  outputs/exp4_singlemod/manifests/val_QSM.csv
  outputs/exp4_singlemod/config_T1.yaml
  outputs/exp4_singlemod/config_QSM.yaml

策略说明:
  - 对于 T1-only: 将原 manifest 中的 `t1` 路径 写入 qsm 列, 并把 t1 列置空。这样现有的 `PDVolDataset` 会把第一通道加载为 T1 (而第二通道变成全0)。
  - 对于 QSM-only: 直接复用 qsm 列, 把 t1 列置空（即正常行为）。
"""
import os
import csv
import argparse
import yaml


def read_csv(path):
    rows = []
    with open(path, 'r') as f:
        rdr = csv.reader(f)
        for r in rdr:
            if not r:
                continue
            rows.append(r)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def make_single_modality_manifests(base_train, base_val, out_manifests_dir):
    train = read_csv(base_train)
    val = read_csv(base_val)

    train_T1 = []
    train_QSM = []
    for r in train:
        # expected row: qsm,t1,aal,label,id
        qsm, t1, aal, label, pid = r[0], r[1] if len(r)>1 else '', r[2] if len(r)>2 else '', r[3] if len(r)>3 else '', r[4] if len(r)>4 else ''
        # T1-only: move t1 -> qsm, empty t1
        train_T1.append([t1, '', aal, label, pid])
        # QSM-only: keep qsm, empty t1
        train_QSM.append([qsm, '', aal, label, pid])

    val_T1 = []
    val_QSM = []
    for r in val:
        qsm, t1, aal, label, pid = r[0], r[1] if len(r)>1 else '', r[2] if len(r)>2 else '', r[3] if len(r)>3 else '', r[4] if len(r)>4 else ''
        val_T1.append([t1, '', aal, label, pid])
        val_QSM.append([qsm, '', aal, label, pid])

    # write
    os.makedirs(out_manifests_dir, exist_ok=True)
    write_csv(os.path.join(out_manifests_dir, 'train_T1.csv'), train_T1)
    write_csv(os.path.join(out_manifests_dir, 'val_T1.csv'), val_T1)
    write_csv(os.path.join(out_manifests_dir, 'train_QSM.csv'), train_QSM)
    write_csv(os.path.join(out_manifests_dir, 'val_QSM.csv'), val_QSM)

    return {
        'train_T1': os.path.join(out_manifests_dir, 'train_T1.csv'),
        'val_T1': os.path.join(out_manifests_dir, 'val_T1.csv'),
        'train_QSM': os.path.join(out_manifests_dir, 'train_QSM.csv'),
        'val_QSM': os.path.join(out_manifests_dir, 'val_QSM.csv'),
    }


def make_configs(base_cfg_path, out_dir, manifests):
    with open(base_cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(out_dir, exist_ok=True)

    cfg_t1 = dict(cfg)
    cfg_t1['data'] = dict(cfg.get('data', {}))
    cfg_t1['data']['train_csv'] = manifests['train_T1']
    cfg_t1['data']['val_csv'] = manifests['val_T1']
    cfg_t1_path = os.path.join(out_dir, 'config_T1.yaml')
    with open(cfg_t1_path, 'w') as f:
        yaml.safe_dump(cfg_t1, f)

    cfg_qsm = dict(cfg)
    cfg_qsm['data'] = dict(cfg.get('data', {}))
    cfg_qsm['data']['train_csv'] = manifests['train_QSM']
    cfg_qsm['data']['val_csv'] = manifests['val_QSM']
    cfg_qsm_path = os.path.join(out_dir, 'config_QSM.yaml')
    with open(cfg_qsm_path, 'w') as f:
        yaml.safe_dump(cfg_qsm, f)

    return cfg_t1_path, cfg_qsm_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str, default='config.yaml')
    parser.add_argument('--out_dir', type=str, default='outputs/exp4_singlemod')
    parser.add_argument('--train_csv', type=str, default='data/train_manifest.csv')
    parser.add_argument('--val_csv', type=str, default='data/val_manifest.csv')
    args = parser.parse_args()

    manifests = make_single_modality_manifests(args.train_csv, args.val_csv, os.path.join(args.out_dir, 'manifests'))
    cfg_t1, cfg_qsm = make_configs(args.base_config, args.out_dir, manifests)

    print('Wrote manifests to:', os.path.join(args.out_dir, 'manifests'))
    print('T1 config:', cfg_t1)
    print('QSM config:', cfg_qsm)


if __name__ == '__main__':
    main()
