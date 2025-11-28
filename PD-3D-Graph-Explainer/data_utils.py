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
                    qsm, t1, aal, label, pid = row[0], row[1], row[2] if row[2]!="" else None, int(row[3]), row[4]
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
            t1  = load_nifti_arr(rec['t1'])
            aal = load_nifti_arr(rec['aal']) if rec['aal'] is not None else None
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

        # light-weight augmentations targeted for small-sample regimes
        if self.transform.get('flip', False) and random.random() < 0.5:
            # apply random flips across any subset of axes for both image channels and segmentation
            axes = []
            if random.random() < 0.5: axes.append(2)  # depth
            if random.random() < 0.5: axes.append(1)  # height
            if random.random() < 0.5: axes.append(0)  # width
            if len(axes) > 0:
                vol = np.flip(vol, axis=[a+1 for a in axes])  # +1 because vol channel is axis 0
                vol = vol.copy()
        noise_sigma = float(self.transform.get('noise_sigma', 0.0))
        if noise_sigma > 0:
            vol[:2] = vol[:2] + np.random.randn(*vol[:2].shape).astype(np.float32) * noise_sigma
            vol = vol.copy()
        vol_t = torch.tensor(vol, dtype=torch.float32)
        sample = {'volume': vol_t, 'aal': aal, 'label': int(label), 'id': pid}
        return sample

def collate_fn(batch):
    return batch
