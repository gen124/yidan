import os
import csv
import random
import numpy as np
import nibabel as nib
from skimage.segmentation import slic
import torch
from torch.utils.data import Dataset

def load_nifti_arr(path):
    img = nib.load(path)
    arr = img.get_fdata()
    return np.asarray(arr).astype(np.float32)

def zscore_normalize(vol, mask=None):
    v = vol.astype(np.float32)
    if mask is not None:
        m = (mask>0)
        if m.sum()>0:
            mean = v[m].mean(); std = v[m].std() if v[m].std()>0 else 1.0
        else:
            mean = v.mean(); std = v.std() if v.std()>0 else 1.0
    else:
        mean = v.mean(); std = v.std() if v.std()>0 else 1.0
    return (v - mean) / std

class PDVolDataset(Dataset):
    def __init__(self, manifest_csv=None, input_size=(192,192,128), mode='train', transform=None, synthetic=False):
        self.items = []
        self.input_size = input_size
        self.transform = transform or {}
        self.synthetic = synthetic
        self.cache = {} # 内存缓存，加速 PatchDataset 的重复访问
        if self.synthetic:
            for i in range(40 if mode=='train' else 10):
                # synthetic entries include an aal field (third channel) -- PDVolDataset will populate
                # a zero segmentation map by default when synthetic=True
                self.items.append({'qsm':None, 't1':None, 'aal':None, 'label': int(i%2), 'id':f"syn_{mode}_{i}"})
        else:
            if manifest_csv is None or not os.path.exists(manifest_csv):
                raise ValueError("manifest_csv missing. Or set synthetic=True for debug.")
            with open(manifest_csv, 'r') as f:
                rdr = csv.reader(f)
                for row in rdr:
                    if len(row) < 5:
                        raise ValueError("manifest csv must have qsm,t1,aal,label,id")
                    qsm, t1, aal, label, pid = row[0], row[1] if row[1]!="" else None, row[2] if row[2]!="" else None, int(row[3]), row[4]
                    # qsm 必须存在；t1/aal 若缺失在加载时用全零替代，避免浪费样本
                    if not os.path.exists(qsm):
                        print(f"[WARN] drop sample {pid}: missing qsm {qsm}")
                        continue
                    self.items.append({'qsm':qsm, 't1':t1, 'aal':aal, 'label':label, 'id':pid})

    def __len__(self): return len(self.items)

    def center_crop_or_pad(self, arr, out_shape):
        a = arr
        if a.ndim != 3:
            raise ValueError("Input volume must be 3D")
        D, H, W = a.shape
        od, oh, ow = out_shape[2], out_shape[0], out_shape[1]
        pad_d = max(0, od - D)
        pad_h = max(0, oh - H)
        pad_w = max(0, ow - W)
        pad = ((pad_d//2, pad_d - pad_d//2),
               (pad_h//2, pad_h - pad_h//2),
               (pad_w//2, pad_w - pad_w//2))
        if any(p>0 for pair in pad for p in pair):
            a = np.pad(a, pad, mode='constant', constant_values=0)
        D2, H2, W2 = a.shape
        sd = (D2 - od)//2; ed = sd + od
        sh = (H2 - oh)//2; eh = sh + oh
        sw = (W2 - ow)//2; ew = sw + ow
        return a[sd:ed, sh:eh, sw:ew]

    def __getitem__(self, idx):
        if idx in self.cache:
            vol, aal, label, pid = self.cache[idx]
        else:
            rec = self.items[idx]
            if self.synthetic:
                D,H,W = self.input_size[2], self.input_size[0], self.input_size[1]
                qsm = np.random.randn(D,H,W).astype(np.float32)
                t1  = np.random.randn(D,H,W).astype(np.float32)
                # synthetic ROI map: zeros (no ROI)
                aal = np.zeros((D,H,W), dtype=np.float32)
                label = rec['label']
                pid = rec['id']
            else:
                qsm = load_nifti_arr(rec['qsm'])
                # t1/aal 允许缺失：若路径为空或不存在则用全零
                t1  = load_nifti_arr(rec['t1']) if (rec['t1'] is not None and os.path.exists(rec['t1'])) else np.zeros_like(qsm, dtype=np.float32)
                aal = load_nifti_arr(rec['aal']) if (rec['aal'] is not None and os.path.exists(rec['aal'])) else None
                qsm = self.center_crop_or_pad(qsm, self.input_size)
                t1  = self.center_crop_or_pad(t1, self.input_size)
                if aal is not None:
                    aal = self.center_crop_or_pad(aal, self.input_size)
                label = rec['label']; pid = rec['id']
            qsm = zscore_normalize(qsm, mask=(aal>0) if aal is not None else None)
            t1  = zscore_normalize(t1, mask=(aal>0) if aal is not None else None)
            # ROI segmentation map should not be z-scored - keep labels / mask values as float channel
            if aal is None:
                aal_chan = np.zeros_like(qsm, dtype=np.float32)
            else:
                aal_chan = aal.astype(np.float32)
            vol = np.stack([qsm, t1, aal_chan], axis=0)
            vol_t = torch.from_numpy(vol).float()
            # 存入缓存 (不含随机增强)
            self.cache[idx] = (vol_t, aal, label, pid)

        vol_t, aal, label, pid = self.cache[idx]
        sample = {'volume': vol_t, 'aal': aal, 'label': int(label), 'id': pid}
        return sample

def collate_fn(batch):
    return batch


class PatchDataset(Dataset):
    """
    从原始3D体积切patch的数据集(用于patch级预训练和迭代重标)
    
    工作流程:
    1. 预训练阶段: 每个3D样本按patch_grid拆成多个patch,所有patch继承患者标签
    2. 迭代重标阶段: 根据模型预测分数更新patch标签(高置信patch重新标注)
    """
    def __init__(self, base_dataset, patch_size=(48,48,32), stride=None, patch_labels=None, mode='pretrain', transform=None):
        """
        Args:
            base_dataset: PDVolDataset实例,提供完整的3D体积
            patch_size: 单个patch的空间大小 (H, W, D)
            stride: 滑动窗口步长,默认=patch_size(无重叠)
            patch_labels: dict {(patient_id, patch_idx): refined_label} 用于迭代重标阶段
            mode: 'pretrain' (用患者标签) 或 'finetune' (用refined标签)
            transform: 增强配置
        """
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.stride = stride if stride else patch_size
        self.patch_labels = patch_labels or {}
        self.mode = mode
        self.transform = transform or {}
        
        # 预计算所有patch的索引: (patient_idx, patch_coords, patient_label)
        self.patch_index = []
        for pt_idx in range(len(base_dataset)):
            sample = base_dataset[pt_idx]
            vol = sample['volume']  # (C, D, H, W)
            patient_label = sample['label']
            patient_id = sample['id']
            
            C, D, H, W = vol.shape
            ph, pw, pd = self.patch_size
            sh, sw, sd = self.stride
            
            # 计算每个维度的patch数量
            n_h = max(1, (H - ph) // sh + 1)
            n_w = max(1, (W - pw) // sw + 1)
            n_d = max(1, (D - pd) // sd + 1)
            
            patch_idx = 0
            for i in range(n_h):
                for j in range(n_w):
                    for k in range(n_d):
                        h_start = i * sh
                        w_start = j * sw
                        d_start = k * sd
                        
                        # 边界检查
                        if h_start + ph > H or w_start + pw > W or d_start + pd > D:
                            continue
                        
                        coords = (h_start, w_start, d_start, ph, pw, pd)
                        
                        # 获取该patch的标签
                        if self.mode == 'finetune' and (patient_id, patch_idx) in self.patch_labels:
                            label = self.patch_labels[(patient_id, patch_idx)]
                        else:
                            label = patient_label  # 预训练阶段使用患者伪标签
                        
                        self.patch_index.append({
                            'patient_idx': pt_idx,
                            'patient_id': patient_id,
                            'patch_idx': patch_idx,
                            'coords': coords,
                            'label': label
                        })
                        patch_idx += 1
    
    def __len__(self):
        return len(self.patch_index)
    
    def __getitem__(self, idx):
        info = self.patch_index[idx]
        pt_idx = info['patient_idx']
        coords = info['coords']
        label = info['label']
        
        # 加载完整体积
        sample = self.base_dataset[pt_idx]
        vol = sample['volume']  # (C, D, H, W)
        
        # 提取patch
        h_s, w_s, d_s, ph, pw, pd = coords
        patch = vol[:, d_s:d_s+pd, h_s:h_s+ph, w_s:w_s+pw]  # (C, pd, ph, pw)
        
        # 在 Patch 级别进行增强，计算量极小
        if self.transform:
            patch_np = patch.numpy()
            if self.transform.get('flip', False) and random.random() < 0.5:
                axes = []
                if random.random() < 0.5: axes.append(1) # depth
                if random.random() < 0.5: axes.append(2) # height
                if random.random() < 0.5: axes.append(3) # width
                if axes:
                    patch_np = np.flip(patch_np, axis=axes).copy()
            
            noise_sigma = float(self.transform.get('noise_sigma', 0.0))
            if noise_sigma > 0:
                patch_np[:2] += np.random.randn(*patch_np[:2].shape).astype(np.float32) * noise_sigma
            
            if self.transform.get('intensity_scale', False) and random.random() < 0.5:
                scale = float(self.transform.get('intensity_scale_range', 0.1))
                mult = 1.0 + np.random.uniform(-scale, scale)
                patch_np[:2] *= mult
            
            if self.transform.get('rotate90', False) and random.random() < float(self.transform.get('rotate90_prob', 0.3)):
                patch_np = np.rot90(patch_np, k=random.choice([1,2,3]), axes=(2,3)).copy()
            
            patch = torch.from_numpy(patch_np)

        return {
            'volume': patch,
            'label': label,
            'patient_id': info['patient_id'],
            'patch_idx': info['patch_idx'],
            'id': f"{info['patient_id']}_patch{info['patch_idx']}"
        }
