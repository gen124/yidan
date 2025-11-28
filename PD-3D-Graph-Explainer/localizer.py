import os
import json
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt

def _normalize(x):
    mi = np.nanmin(x); ma = np.nanmax(x)
    if ma - mi == 0:
        return np.zeros_like(x)
    return (x - mi) / (ma - mi)

def auto_localize(npz_path, out_dir, t1_path=None, qsm_path=None, aal_path=None, fuse_alpha=0.5, threshold='percentile', perc=95, min_size=10):
    """
    Automatic localization of abnormal regions using fused CAM + GNN importance maps.

    Inputs:
      npz_path: path to voxel_importances.npz (cam_orig, gnn_orig, ...)
      out_dir: folder where localization outputs will be written
      t1_path/qsm_path: optional original images for visualization overlays
      aal_path: optional ROI atlas (will add ROI labels for localized regions)
      fuse_alpha: weight of CAM in fusion (0..1), final = alpha*cam + (1-alpha)*gnn
      threshold: 'percentile' or float (0..1) threshold for fused map
      perc: percentile if threshold == 'percentile'
      min_size: minimum voxel count for a region to be kept

    Outputs:
      - out_dir/localization.json (list of detected regions with centroid, bbox, mean scores)
      - out_dir/fused_map.nii.gz
      - out_dir/region_{i}_mask.nii.gz for each region
      - optional overlays PNGs: out_dir/overlay_slice_{z}.png
    """
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path)
    if 'cam_orig' in data:
        cam = data['cam_orig']
    else:
        cam = data['cam_feat']
    gnn = data.get('gnn_orig') if 'gnn_orig' in data else data.get('gnn_feat')
    if gnn is None:
        raise ValueError('gnn map not found in npz')

    cam_n = _normalize(cam)
    gnn_n = _normalize(gnn)
    fused = fuse_alpha * cam_n + (1.0 - fuse_alpha) * gnn_n

    # threshold fused map
    if isinstance(threshold, str) and threshold == 'percentile':
        th = np.percentile(fused, perc)
    else:
        th = float(threshold)

    mask = fused >= th

    # connected components
    labeled, ncomp = ndimage.label(mask)

    regions = []
    for lab in range(1, ncomp+1):
        comp = (labeled == lab)
        size = int(comp.sum())
        if size < min_size:
            continue
        coords = np.array(np.nonzero(comp)).T  # (N, 3) coords (z,y,x)
        zmin, ymin, xmin = coords.min(axis=0).tolist()
        zmax, ymax, xmax = coords.max(axis=0).tolist()
        bbox = [int(zmin), int(ymin), int(xmin), int(zmax), int(ymax), int(xmax)]
        centroid_vox = coords.mean(axis=0).tolist()
        mean_fused = float(fused[comp].mean())
        mean_cam = float(cam[comp].mean())
        mean_gnn = float(gnn[comp].mean())

        roi_overlaps = None
        if aal_path is not None and os.path.exists(aal_path):
            aal = nib.load(aal_path).get_fdata().astype(int)
            # map component in feature-space -> needs to be same shape as aal
            # assume aal already in original space; mask is original space sized
            unique_labels, counts = np.unique(aal[comp], return_counts=True)
            # ignore background label 0
            pairs = [(int(l), int(c)) for l, c in zip(unique_labels, counts) if l != 0]
            roi_overlaps = sorted(pairs, key=lambda x: -x[1])

        regions.append({
            'label': int(lab),
            'size': size,
            'bbox': bbox,
            'centroid_voxel': [float(x) for x in centroid_vox],
            'mean_fused': mean_fused,
            'mean_cam': mean_cam,
            'mean_gnn': mean_gnn,
            'roi_overlaps': roi_overlaps,
        })

    # Save fused map as NIfTI (no affine known) - write with identity affine
    fused_nifti = nib.Nifti1Image(fused.astype(np.float32), np.eye(4))
    nib.save(fused_nifti, os.path.join(out_dir, 'fused_map.nii.gz'))

    # Save region masks and optionally overlay slices
    for r in regions:
        m = (labeled == r['label']).astype(np.uint8)
        nib.save(nib.Nifti1Image(m, np.eye(4)), os.path.join(out_dir, f"region_{r['label']}_mask.nii.gz"))

    # write JSON summary
    with open(os.path.join(out_dir, 'localization.json'), 'w') as f:
        json.dump({'n_regions': len(regions), 'regions': regions}, f, indent=2)

    # optionally create a few 2D overlay slices on T1 or QSM
    img = None
    if t1_path and os.path.exists(t1_path):
        img = nib.load(t1_path).get_fdata().astype(np.float32)
    elif qsm_path and os.path.exists(qsm_path):
        img = nib.load(qsm_path).get_fdata().astype(np.float32)

    if img is not None:
        # create per-slice overlays (max 8 slices) centered on detected regions
        slices_to_plot = set()
        for r in regions:
            zc = int(round(r['centroid_voxel'][0]))
            slices_to_plot.update(range(max(0, zc-1), min(img.shape[0], zc+2)))
        slices_to_plot = sorted(list(slices_to_plot))[:8]
        for z in slices_to_plot:
            plt.figure(figsize=(8,6))
            plt.imshow(img[z,:,:].T, cmap='gray', origin='lower')
            plt.imshow(fused[z,:,:].T, cmap='hot', alpha=0.5, origin='lower')
            plt.title(f'overlay z={z}')
            plt.axis('off')
            plt.savefig(os.path.join(out_dir, f'overlay_slice_{z}.png'))
            plt.close()

    return {'n_regions': len(regions), 'regions': regions, 'fused_map': os.path.join(out_dir, 'fused_map.nii.gz'), 'localization_json': os.path.join(out_dir, 'localization.json')}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--out', default='localize_out')
    p.add_argument('--t1', default=None)
    p.add_argument('--qsm', default=None)
    p.add_argument('--aal', default=None)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--perc', type=float, default=95.0)
    args = p.parse_args()
    res = auto_localize(args.npz, args.out, t1_path=args.t1, qsm_path=args.qsm, aal_path=args.aal, fuse_alpha=args.alpha, perc=args.perc)
    print('Localized', res['n_regions'], 'regions ->', res['localization_json'])
