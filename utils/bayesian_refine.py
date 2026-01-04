"""Simplified bayesian_refine helpers.

This module provides lightweight, deterministic helpers used by the
training and inference pipelines. Implementations are intentionally
simple to restore import compatibility and enable quick tests.
"""

from typing import List, Tuple, Dict

import numpy as np


def _clamp_prob_array(x):
    a = np.asarray(x, dtype=float)
    a = np.clip(a, 0.0, 1.0)
    return a


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def subject_conditional_soft_refine(
    patch_scores: Dict[Tuple, List[float]] | Dict[Tuple, float],
    subject_scores: Dict,
    alpha: float = 1.5,
    freeze_high_conf: bool = True,
    high_conf_threshold: float = 0.9,
    low_conf_threshold: float = 0.1,
) -> Tuple[Dict[Tuple, float], Dict]:
    """Refine per-patch probabilities conditioned on subject-level scores.

    Accepts either per-subject lists `patch_scores[sid] = [p1,p2,...]` or
    a flat dict `patch_scores[(sid, idx)] = p`.

    Returns (refined_flat, stats).
    """
    # Normalize to flat dict
    flat = {}
    if all(isinstance(v, list) for v in patch_scores.values()):
        for sid, lst in patch_scores.items():
            for i, p in enumerate(lst):
                flat[(sid, i)] = float(p)
    else:
        flat = {k: float(v) for k, v in patch_scores.items()}

    refined = {}
    stats = {"frozen_high_conf": 0, "soft_updated": 0}

    for (sid, idx), p in flat.items():
        subj_p = float(subject_scores.get(sid, 0.5))

        if freeze_high_conf and (p >= high_conf_threshold or p <= low_conf_threshold):
            refined[(sid, idx)] = p
            stats["frozen_high_conf"] += 1
            continue

        # simple logit-ish update: move p toward subj_p
        try:
            lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            ls = np.log(np.clip(subj_p, 1e-6, 1 - 1e-6) / (1 - np.clip(subj_p, 1e-6, 1 - 1e-6)))
            updated = _sigmoid(lp + alpha * (ls - 0.0))
        except Exception:
            updated = float((p + alpha * subj_p) / (1.0 + alpha))

        refined[(sid, idx)] = float(np.clip(updated, 0.0, 1.0))
        stats["soft_updated"] += 1

    return refined, stats


def spatial_soft_refine(
    patch_probs: Dict[Tuple, float],
    patch_neighbors: Dict[Tuple, List[Tuple]],
    lambda_spatial: float = 0.3,
    n_iters: int = 2,
) -> Dict[Tuple, float]:
    """Smooth patch probabilities over neighbor graph.

    `patch_neighbors` maps a patch key to a list of neighbor keys.
    """
    probs = {k: float(v) for k, v in patch_probs.items()}
    for _ in range(max(1, int(n_iters))):
        new_probs = {}
        for key, p in probs.items():
            neighs = patch_neighbors.get(key, [])
            if not neighs:
                new_probs[key] = p
                continue
            neigh_vals = [probs.get(nk, p) for nk in neighs]
            neigh_mean = float(np.mean(neigh_vals))
            new_probs[key] = float((1 - lambda_spatial) * p + lambda_spatial * neigh_mean)
        probs = new_probs
    # clamp
    return {k: float(np.clip(v, 0.0, 1.0)) for k, v in probs.items()}


def build_patch_neighbors(patch_index_map: Dict[Tuple, Tuple[float, float, float]], radius: float = 1.0):
    """Build neighbor lists based on simple L2 distance within `radius`.

    patch_index_map: dict {(sid, pid): (x,y,z)}
    returns: dict {(sid, pid): [neighbor_keys]}
    """
    keys = list(patch_index_map.keys())
    coords = np.array([patch_index_map[k] for k in keys], dtype=float)
    N = len(keys)
    neighbors = {k: [] for k in keys}
    if N <= 1:
        return neighbors

    for i in range(N):
        for j in range(i + 1, N):
            if keys[i][0] != keys[j][0]:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d <= radius:
                neighbors[keys[i]].append(keys[j])
                neighbors[keys[j]].append(keys[i])
    return neighbors


def mil_pooling_from_patch_scores(patch_scores: Dict[Tuple, float], method: str = "topk_mean", topk: int = 5) -> Dict:
    """Aggregate flat patch-level scores {(sid, pid): score} to {sid: pooled}.

    Supported methods: 'max', 'mean', 'topk_mean'.
    """
    subject_pools = {}
    for (sid, _), score in patch_scores.items():
        subject_pools.setdefault(sid, []).append(float(score))

    out = {}
    for sid, scores in subject_pools.items():
        arr = np.array(scores, dtype=float)
        if arr.size == 0:
            out[sid] = 0.0
            continue
        if method == "max":
            out[sid] = float(np.max(arr))
        elif method == "mean":
            out[sid] = float(np.mean(arr))
        elif method == "topk_mean":
            k = max(1, min(int(topk), arr.size))
            out[sid] = float(np.mean(np.sort(arr)[-k:]))
        else:
            raise ValueError(f"Unknown pooling method: {method}")
    return out


__all__ = [
    "subject_conditional_soft_refine",
    "spatial_soft_refine",
    "build_patch_neighbors",
    "mil_pooling_from_patch_scores",
]

