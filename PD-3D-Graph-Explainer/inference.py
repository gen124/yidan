import os
import torch
import numpy as np
from models import ResNet3D
from data_utils import load_nifti_arr, zscore_normalize
from graph_builder import build_roi_graph, build_slice_graph, build_supervoxel_graph
from skimage.transform import resize as sk_resize
import json
from explainer import fit_tiny_gnn_and_explain, path_pdm
import nibabel as nib
import matplotlib.pyplot as plt
import yaml

def load_cfg(path='config.yaml'):
    with open(path,'r') as f: return yaml.safe_load(f)

def register_forward_hook(model, layer_name='layer4'):
    features = {}
    def get_hook(name):
        def hook(module, input, output):
            # keep output tensor (on device) so gradients are accessible; request PyTorch to retain grad
            # even if it's not a leaf tensor (non-leaf grads are normally not retained).
            try:
                output.retain_grad()
            except Exception:
                pass
            features['feat'] = output
        return hook
    mod = dict(model.named_modules()).get(layer_name, None)
    if mod is None:
        mod = list(model.named_modules())[-1][1]
    handle = mod.register_forward_hook(get_hook(layer_name))
    return handle, features

def grad_cam_3d(model, input_tensor, target_class=None, layer_name='layer4', device='cuda'):
    model.eval()
    input_tensor = input_tensor.to(device)
    handle, features = register_forward_hook(model, layer_name=layer_name)
    model.zero_grad()
    out = model(input_tensor)
    if isinstance(out, tuple):
        logit = out[0]
    else:
        logit = out
    prob = torch.sigmoid(logit)
    if target_class is None:
        score = prob
    else:
        score = prob[0] if isinstance(prob, torch.Tensor) else torch.tensor(prob)
    score.backward(retain_graph=True)
    feat = features.get('feat')
    if feat is None:
        handle.remove(); raise RuntimeError("feature hook failed")
    grads = feat.grad if feat.requires_grad else None
    if grads is None:
        handle.remove()
        input_tensor.requires_grad_(True)
        handle, features = register_forward_hook(model, layer_name=layer_name)
        out = model(input_tensor)
        prob = torch.sigmoid(out)
        prob.backward()
        feat = features['feat']; grads = feat.grad
    weights = grads.mean(dim=[2,3,4], keepdim=True)
    cam = (weights * feat).sum(dim=1).squeeze(0)
    cam = torch.relu(cam)
    cam_np = cam.detach().cpu().numpy()
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-12)
    handle.remove()
    return cam_np, prob.item()

def run_inference_and_explain(model_ckpt, qsm_path, t1_path, aal_path=None, cfg_path='config.yaml', out_dir='./explain_out'):
    cfg = load_cfg(cfg_path)
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    model = ResNet3D(in_channels=cfg['model']['in_channels'], base_channels=cfg['model']['base_channels']).to(device)
    ck = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck)
    model.eval()
    qsm = load_nifti_arr(qsm_path)
    t1  = load_nifti_arr(t1_path)
    qsm_n = zscore_normalize(qsm)
    t1_n  = zscore_normalize(t1)
    # prepare aal channel - try to align shape with images
    if aal_path and os.path.exists(aal_path):
        # reuse `aal` loaded / aligned earlier (if present). If for some reason it's missing, reload.
        try:
            _ = aal
        except NameError:
            aal = load_nifti_arr(aal_path)
        # if shapes mismatch, center-crop or pad aal to match qsm
        if aal.shape != qsm.shape:
            def center_crop_or_pad(arr, out_shape):
                a = arr
                D, H, W = a.shape
                od, oh, ow = out_shape
                pad_d = max(0, od - D); pad_h = max(0, oh - H); pad_w = max(0, ow - W)
                pad = ((pad_d//2, pad_d - pad_d//2), (pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2))
                if any(p>0 for pair in pad for p in pair):
                    a = np.pad(a, pad, mode='constant', constant_values=0)
                D2, H2, W2 = a.shape
                sd = (D2 - od)//2; ed = sd + od
                sh = (H2 - oh)//2; eh = sh + oh
                sw = (W2 - ow)//2; ew = sw + ow
                return a[sd:ed, sh:eh, sw:ew]
            aal = center_crop_or_pad(aal, qsm.shape)
        aal_n = aal.astype(np.float32)
    else:
        aal_n = np.zeros_like(qsm, dtype=np.float32)
    # stack three channels: QSM, T1, and ROI segmentation map
    vol = np.stack([qsm_n, t1_n, aal_n], axis=0)
    x = torch.tensor(vol[np.newaxis,...], dtype=torch.float32).to(device)
    cam3d, prob = grad_cam_3d(model, x, target_class=None, layer_name='layer4', device=device)
    print("Pred prob:", prob)
    handle, feat_container = register_forward_hook(model, layer_name='layer4')
    with torch.no_grad():
        _ = model(x)
    feat = feat_container['feat'].squeeze(0).cpu().numpy()
    handle.remove()
    if aal_path and os.path.exists(aal_path):
        aal = load_nifti_arr(aal_path)
        # resample aal (nearest) to feature-map spatial dims (Df,Hf,Wf)
        Df, Hf, Wf = feat.shape[1:]
        aal_feat = sk_resize(aal, output_shape=(Df, Hf, Wf), order=0, preserve_range=True, anti_aliasing=False).astype(aal.dtype)
        nodes, feats, edges, centers, labels = build_roi_graph(feat, aal_feat)
        graph_type = 'roi'
    else:
        nodes, feats, edges, centers = build_slice_graph(feat)
        graph_type = 'slice'
    os.makedirs(out_dir, exist_ok=True)
    node_scores_cam = []
    D,H,W = cam3d.shape
    for c in centers:
        z,y,xp = int(round(c[0])), int(round(c[1])), int(round(c[2]))
        z = max(0,min(D-1,z)); y=max(0,min(H-1,y)); xp=max(0,min(W-1,xp))
        node_scores_cam.append(float(cam3d[z,y,xp]))
    node_scores_cam = np.array(node_scores_cam, dtype=np.float32)
    node_imp, edge_imp = fit_tiny_gnn_and_explain(nodes, feats, edges, prob, device=device, n_runs=cfg['explainer'].get('n_runs',1), noise_std=float(cfg['explainer'].get('noise_std',0.0)))
    paths = path_pdm(node_imp, edges, topk=cfg['explainer']['topk_path'])
    import csv
    with open(os.path.join(out_dir, "node_importances.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['node','cam_score','gnn_score'])
        for i,n in enumerate(nodes):
            w.writerow([n, float(node_scores_cam[i]), float(node_imp[i])])
    import matplotlib.pyplot as plt
    top_idx = np.argsort(-node_imp)[:10]
    plt.figure(figsize=(10,4)); plt.bar(range(len(top_idx)), node_imp[top_idx]); plt.xticks(range(len(top_idx)), [nodes[i] for i in top_idx], rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_nodes.png"))
    print("Top paths (node idx):", paths)

    # --- Build voxel-level importance maps (feature-space -> original-space) ---
    Df, Hf, Wf = cam3d.shape
    orig_D, orig_H, orig_W = qsm.shape

    def upsample_to_original(arr_feat, out_shape=(orig_D, orig_H, orig_W), order=1):
        return sk_resize(arr_feat, output_shape=out_shape, order=order, preserve_range=True, anti_aliasing=False)

    # CAM is already feature-space; upsample to original image resolution
    cam_orig = upsample_to_original(cam3d, out_shape=(orig_D, orig_H, orig_W), order=1)

    # initialize gnn voxel map (feature-space). We'll support ROI / slice / supervoxel
    gnn_feat_map = np.zeros((Df, Hf, Wf), dtype=np.float32)
    node_infos = []
    # if ROI graph (labels available)
    if 'labels' in locals() and labels is not None:
        labels_feat = aal_feat.astype(int)
        for i, lab in enumerate(labels):
            mask = (labels_feat == int(lab))
            if np.any(mask):
                gnn_feat_map[mask] = float(node_imp[i])
                # compute original-space centroid
                centroid_feat = centers[i]
                # scale to original volume coords
                zc = int(round(centroid_feat[0] * (orig_D / Df)))
                yc = int(round(centroid_feat[1] * (orig_H / Hf)))
                xc = int(round(centroid_feat[2] * (orig_W / Wf)))
                node_infos.append({'idx': i, 'node': nodes[i], 'type': 'roi', 'label': int(lab), 'centroid_feat': [float(x) for x in centroid_feat], 'centroid_orig': [zc,yc,xc], 'cam_score': float(node_scores_cam[i]), 'gnn_score': float(node_imp[i])})
    elif nodes and nodes[0].startswith('sv_') and len(feats) > 0:
        # supervoxel case - build_supervoxel_graph returns labels variable but for consistency ensure labels exist
        # We attempted to use supervoxels via build_supervoxel_graph in other codepaths - here label mapping variable would be labels
        try:
            labels_sv
        except NameError:
            labels_sv = None
        # If labels array exists from build_supervoxel_graph earlier
        if 'labels' in locals() and labels is not None:
            sv_labels = labels
            for i, lab in enumerate(sv_labels):
                mask = (labels == int(lab))
                if np.any(mask):
                    gnn_feat_map[mask] = float(node_imp[i])
                    centroid_feat = centers[i]
                    zc = int(round(centroid_feat[0] * (orig_D / Df)))
                    yc = int(round(centroid_feat[1] * (orig_H / Hf)))
                    xc = int(round(centroid_feat[2] * (orig_W / Wf)))
                    node_infos.append({'idx': i, 'node': nodes[i], 'type': 'supervoxel', 'label': int(lab), 'centroid_feat': [float(x) for x in centroid_feat], 'centroid_orig': [zc,yc,xc], 'cam_score': float(node_scores_cam[i]), 'gnn_score': float(node_imp[i])})
    else:
        # slice-level mapping: nodes are slice_z
        for i in range(len(nodes)):
            z = int(i)
            if z < Df:
                gnn_feat_map[z,:,:] = float(node_imp[i])
                # compute centroid: (z, Hf//2, Wf//2)
                centroid_feat = centers[i]
                zc = int(round(centroid_feat[0] * (orig_D / Df)))
                yc = int(round(centroid_feat[1] * (orig_H / Hf)))
                xc = int(round(centroid_feat[2] * (orig_W / Wf)))
                node_infos.append({'idx': i, 'node': nodes[i], 'type': 'slice', 'label': None, 'centroid_feat': [float(x) for x in centroid_feat], 'centroid_orig': [zc,yc,xc], 'cam_score': float(node_scores_cam[i]), 'gnn_score': float(node_imp[i])})

    # upsample gnn_feat_map to original space (linear interpolation)
    gnn_orig = upsample_to_original(gnn_feat_map, out_shape=(orig_D, orig_H, orig_W), order=1)

    # Write outputs: per-voxel importance maps and JSON summary
    np.savez_compressed(os.path.join(out_dir, 'voxel_importances.npz'), cam_feat=cam3d, cam_orig=cam_orig, gnn_feat=gnn_feat_map, gnn_orig=gnn_orig)
    # make paths JSON-serializable (convert numpy/np.float to native types)
    ser_paths = [[float(c), [int(x) for x in p]] for c,p in paths]
    with open(os.path.join(out_dir, 'top_nodes.json'), 'w') as jf:
        json.dump({'prob': float(prob), 'nodes': node_infos, 'paths': ser_paths}, jf, indent=2)

    return {'prob':prob, 'nodes':nodes, 'node_imp':node_imp, 'cam_scores':node_scores_cam, 'paths':paths, 'voxel_importances': os.path.join(out_dir, 'voxel_importances.npz'), 'top_nodes_json': os.path.join(out_dir,'top_nodes.json')}
