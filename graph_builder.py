import numpy as np
from scipy.spatial.distance import cdist
from skimage.segmentation import slic

def pool_feat_in_mask(feat_np, mask_bool):
    C = feat_np.shape[0]
    if mask_bool.sum()==0:
        return np.zeros(C, dtype=np.float32)
    vals = feat_np[:, mask_bool]
    return vals.mean(axis=1)

def build_roi_graph(feat_np, aal_map):
    labels = np.unique(aal_map)
    labels = labels[labels>0]
    nodes = []; feats = []; centroids = []
    for lab in labels:
        mask = (aal_map==lab)
        f = pool_feat_in_mask(feat_np, mask)
        nodes.append(f"roi_{int(lab)}"); feats.append(f)
        inds = np.array(np.where(mask)).T
        if inds.shape[0]==0:
            centroids.append((0,0,0))
        else:
            centroids.append(tuple(np.mean(inds, axis=0).tolist()))
    feats = np.stack(feats, axis=0).astype(np.float32)
    cent = np.array(centroids)
    if len(cent) <= 1:
        return nodes, feats, [], centroids, labels
    dists = cdist(cent, cent)
    sigma = max(1e-6, np.median(dists[dists>0]))
    weights = np.exp(-(dists**2)/(2*sigma**2 + 1e-12))
    np.fill_diagonal(weights, 0.0)
    edges = []
    k = min(6, len(nodes)-1)
    for i in range(len(nodes)):
        idxs = np.argsort(-weights[i])[:k]
        for j in idxs:
            if weights[i,j] > 0:
                edges.append((i,j, float(weights[i,j])))
    return nodes, feats, edges, centroids, labels

def build_slice_graph(feat_np):
    C,D,H,W = feat_np.shape
    nodes = [f"slice_{z}" for z in range(D)]
    feats = []
    centroids = []
    for z in range(D):
        mask = np.zeros((D,H,W), dtype=bool); mask[z,:,:] = True
        feats.append(pool_feat_in_mask(feat_np, mask))
        centroids.append((z, H//2, W//2))
    feats = np.stack(feats, axis=0).astype(np.float32)
    edges = []
    for z in range(D-1):
        edges.append((z, z+1, 1.0))
    return nodes, feats, edges, centroids

def build_supervoxel_graph(feat_np, vol_np, n_segments=200, compactness=0.1):
    labels = slic(vol_np, n_segments=n_segments, compactness=compactness, multichannel=False, start_label=0)
    labs = np.unique(labels)
    nodes = []; feats = []; centroids = []
    for lab in labs:
        mask = (labels==lab)
        f = pool_feat_in_mask(feat_np, mask)
        nodes.append(f"sv_{int(lab)}"); feats.append(f)
        inds = np.array(np.where(mask)).T
        centroids.append(tuple(np.mean(inds, axis=0).tolist()))
    feats = np.stack(feats, axis=0).astype(np.float32)
    cent = np.array(centroids)
    if len(cent) <= 1:
        return nodes, feats, [], centroids, labels
    dists = cdist(cent, cent)
    sigma = max(1e-6, np.median(dists[dists>0]))
    weights = np.exp(-(dists**2)/(2*sigma**2 + 1e-12))
    np.fill_diagonal(weights, 0.0)
    edges = []
    k = min(8, len(nodes)-1)
    for i in range(len(nodes)):
        idxs = np.argsort(-weights[i])[:k]
        for j in idxs:
            if weights[i,j] > 0:
                edges.append((i,j,float(weights[i,j])))
    return nodes, feats, edges, centroids, labels
