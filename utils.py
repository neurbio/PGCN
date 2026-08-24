# --------------------------
# basic utils
# --------------------------
import os
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components

def to_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def rmse_mae(X_true, X_pred, mask):
    diff = (X_pred - X_true)[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae  = float(np.mean(np.abs(diff)))
    return rmse, mae

def make_holdout_masks_pos_zero(X_true, prob_pos=0.01, prob_zero=0.001, seed=0):
    rng = np.random.default_rng(seed)
    pos = (X_true > 0)
    zero = ~pos
    holdout_pos  = pos  & (rng.random(X_true.shape) < prob_pos)
    holdout_zero = zero & (rng.random(X_true.shape) < prob_zero)
    return holdout_pos, holdout_zero


def graph_stats(A):
    deg = np.array(A.astype(bool).sum(axis=1)).ravel()
    n_comp, labels = connected_components(A, directed=False, connection="weak")
    sizes = np.bincount(labels)
    sizes_sorted = np.sort(sizes)[::-1]
    return {
        "n": A.shape[0],
        "nnz": int(A.nnz),
        "min_deg": int(deg.min()),
        "mean_deg": float(deg.mean()),
        "max_deg": int(deg.max()),
        "n_components": int(n_comp),
        "largest_component": int(sizes_sorted[0]),
        "top10_component_sizes": sizes_sorted[:10].tolist(),
    }