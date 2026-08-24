# ccg_vgcn.py
# ------------------------------------------------------------
# Build cell-cell graph from X (Pearson -> shrink -> ADMM precision -> partial corr -> mutual topK)
# Then train VGCN to denoise features and reconstruct X.
##
# Example:
#   python ccg_vgcn.py --mode vgcn --topK 20 --apply_mask --mask_prob 0.01 --mask_seed 0
# ------------------------------------------------------------

import os, time, argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix,issparse
from sklearn.decomposition import TruncatedSVD
import scanpy as sc

from ADMM_surrogate import admm_sparse_precision
from train import vgcn_denoise_X
from utils import to_dense, ensure_dir, rmse_mae, make_holdout_masks_pos_zero, graph_stats

def _sample_X(adata, max_cells=300, max_genes=2000, seed=0):
    rng = np.random.default_rng(seed)
    n_obs, n_vars = adata.n_obs, adata.n_vars

    obs_idx = np.arange(n_obs)
    var_idx = np.arange(n_vars)

    if n_obs > max_cells:
        obs_idx = rng.choice(n_obs, size=max_cells, replace=False)
    if n_vars > max_genes:
        var_idx = rng.choice(n_vars, size=max_genes, replace=False)

    X = adata.X[obs_idx][:, var_idx]
    if issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    return X.astype(np.float64)

def _is_integer_like(X, tol=1e-6):
    return np.all(np.abs(X - np.rint(X)) < tol)

def inspect_preprocessing(adata, seed=0):
    """
    Heuristic check:
    - logged: from metadata first, then matrix pattern
    - normalized: estimated from per-cell sums
    """
    Xs = _sample_X(adata, seed=seed)

    nonneg = np.min(Xs) >= -1e-8
    xmax = float(np.max(Xs))
    integer_like = _is_integer_like(Xs)

    # ----- log1p check -----
    log1p_meta = "log1p" in adata.uns
    # heuristic: logged data are usually non-integer, non-negative, and not huge
    logged_heuristic = nonneg and (not integer_like) and (xmax < 30)

    logged = log1p_meta or logged_heuristic

    # ----- normalization check -----
    # if logged, roughly reverse first for library-size inspection
    X_for_sum = np.expm1(Xs) if logged else Xs
    cell_sums = X_for_sum.sum(axis=1)
    mean_sum = float(np.mean(cell_sums))
    cv_sum = float(np.std(cell_sums) / (mean_sum + 1e-8))

    # heuristic:
    # normalized data usually have more similar library sizes across cells
    normalized_heuristic = cv_sum < 0.30

    # some metadata hints
    norm_meta = (
        "normalize_total" in adata.uns
        or "size_factors" in adata.obs.columns
    )

    normalized = norm_meta or normalized_heuristic

    info = {
        "logged": logged,
        "normalized": normalized,
        "log1p_meta": log1p_meta,
        "norm_meta": norm_meta,
        "integer_like": integer_like,
        "nonneg": nonneg,
        "xmax": xmax,
        "mean_cell_sum_est": mean_sum,
        "cv_cell_sum_est": cv_sum,
    }
    return info

def ensure_normalized_log1p(adata, target_sum=1e4, seed=0, verbose=True):
    """
    If already normalized + log1p: keep as is.
    Otherwise preprocess.
    """
    info = inspect_preprocessing(adata, seed=seed)

    if verbose:
        print("\n[check preprocessing]")
        for k, v in info.items():
            print(f"  {k}: {v}")

    already_ok = info["logged"] and info["normalized"]

    if already_ok:
        if verbose:
            print("\n[status] Detected normalized + log1p. Use adata.X directly.")
        return adata, info

    adata = adata.copy()

    # Prefer raw counts layer if available
    if "counts" in adata.layers:
        if verbose:
            print("\n[preprocess] Using adata.layers['counts'] as raw counts.")
        adata.X = adata.layers["counts"].copy()

    else:
        # if data already look logged, reverse first
        if info["logged"]:
            if verbose:
                print("\n[preprocess] log1p detected but normalization not confirmed.")
                print("[preprocess] Reversing with expm1, then normalize_total + log1p.")
            if issparse(adata.X):
                adata.X = adata.X.tocsr(copy=True)
                adata.X.data = np.expm1(adata.X.data)
            else:
                adata.X = np.expm1(np.asarray(adata.X, dtype=np.float64))

    # normalize + log1p
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    if verbose:
        print(f"[preprocess] Done: normalize_total(target_sum={target_sum}) + log1p")

    return adata, info
# --------------------------
# correlation / precision graph
# --------------------------
def pearson_corr_rows(X, eps=1e-12, dtype=np.float32):
    """
    Pearson correlation between rows (cells) of X: returns (n,n) dense.
    X: (n_cells, n_features)
    """
    X = np.asarray(X, dtype=dtype)
    Xm = X - X.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(Xm, axis=1, keepdims=True) + eps
    Xn = Xm / denom
    S = Xn @ Xn.T
    np.fill_diagonal(S, 1.0)
    np.clip(S, -1.0, 1.0, out=S)
    return S

def shrink_corr(S, alpha=1e-3):
    S2 = (1.0 - alpha) * S
    S2 = S2.copy()
    np.fill_diagonal(S2, np.diag(S2) + alpha)
    return S2

def partial_corr_from_precision(Theta, eps=1e-12):
    """
    rho_ij = -Theta_ij / sqrt(Theta_ii * Theta_jj)
    returns W = |rho|, symmetric, diag=0
    """
    Theta = np.asarray(Theta, dtype=np.float64)
    d = np.diag(Theta).copy()
    d = np.maximum(d, eps)
    inv_sqrt_d = 1.0 / np.sqrt(d)

    R = -Theta * (inv_sqrt_d[:, None] * inv_sqrt_d[None, :])
    np.fill_diagonal(R, 0.0)
    W = np.abs(R)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W

def mutual_topk_graph(W, K=20, keep_weights=True):
    """
    W: (n,n) dense nonnegative weights, diag=0
    returns csr symmetric adjacency
    """
    n = W.shape[0]
    K = int(min(K, n - 1))

    idx = np.argpartition(-W, K, axis=1)[:, :K]  # unsorted topK
    rows = np.repeat(np.arange(n), K)
    cols = idx.reshape(-1)

    data_bool = np.ones(rows.shape[0], dtype=np.int8)
    N_dir = csr_matrix((data_bool, (rows, cols)), shape=(n, n))
    N_mut = N_dir.multiply(N_dir.T)

    if keep_weights:
        r, c = N_mut.nonzero()
        w = W[r, c]
        A = csr_matrix((w, (r, c)), shape=(n, n))
        A = 0.5 * (A + A.T)
    else:
        A = 0.5 * (N_mut.astype(np.float64) + N_mut.T.astype(np.float64))

    A.setdiag(0.0)
    A.eliminate_zeros()
    return A

def build_cellcell_graph_admm(
    X_feat,
    topK=20,
    shrink_alpha=1e-3,
    gl_lam=1e-3,
    gl_rho=1e-2,
    admm_max_iter=500,
    zero_diag=False
):
    """
    X_feat: (cells, features) used to compute cell-cell correlation
    Returns A (csr)
    """
    t0 = time.time()
    S = pearson_corr_rows(X_feat, dtype=np.float32)
    S_alpha = shrink_corr(S.astype(np.float64), alpha=shrink_alpha)

    t1 = time.time()
    out = admm_sparse_precision(
        S_alpha, lam=gl_lam, rho=gl_rho,
        max_iter=admm_max_iter, reltol=1e-4, abstol=1e-5,
        penalize_diag=False, zero_diag=zero_diag,
        return_history=False
    )
    # admm returns (Theta, Aout) in your implementation
    Theta = out[0]
    # Aout  = out[1]
    print(f"[ADMM] time={time.time()-t1:.2f}s  Theta shape={Theta.shape}")

    W = partial_corr_from_precision(Theta)
    A = mutual_topk_graph(W, K=topK, keep_weights=True)

    print(f"[GRAPH] build time={time.time()-t0:.2f}s  stats={graph_stats(A)}")
    return A


#
# main
#
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=r"E:\python\GRNC\data\MEScounts10426.h5ad")
    p.add_argument("--out_dir", type=str, default=r".\data")
    p.add_argument("--topK", type=int, default=20)
    p.add_argument("--shrink_alpha", type=float, default=1e-3)
    p.add_argument("--gl_lam", type=float, default=1e-3)
    p.add_argument("--gl_rho", type=float, default=1e-2)
    p.add_argument("--admm_max_iter", type=int, default=500)

    # feature choice for graph and VGCN
    p.add_argument("--use_svd_feat", action="store_true",
                   help="If set: use SVD features Z for graph+VGCN; else use raw X (cells x genes) features")
    p.add_argument("--k_use", type=int, default=200, help="SVD rank if --use_svd_feat")

    # VGCN training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden1", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--x_drop_prob", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    # strict mask
    p.add_argument("--apply_mask", action="store_true")
    p.add_argument("--mask_prob", type=float, default=0.01)
    p.add_argument("--mask_seed", type=int, default=0)

    args = p.parse_args()
    ensure_dir(args.out_dir)

    print("Loading:", args.input)
    adata = sc.read(args.input)
    adata, prep_info = ensure_normalized_log1p(adata, target_sum=1e4, seed=0, verbose=True)    # validate first
    X_true = to_dense(adata.X).astype(np.float64)
    print("X shape:", X_true.shape, "min/max:", float(X_true.min()), float(X_true.max()))

    # strict masking
    holdout_pos = holdout_zero = None
    if args.apply_mask:
        holdout_pos, holdout_zero = make_holdout_masks_pos_zero(
            X_true.astype(np.float32),
            prob_pos=args.mask_prob,
            prob_zero=min(args.mask_prob, 0.005),  # keep small!
            seed=args.mask_seed
        )
        holdout = holdout_pos | holdout_zero  # mask both
        X_in = X_true.copy()
        X_in[holdout] = 0.0
        print(
            f"[MASK] holdout_pos={int(holdout_pos.sum())} holdout_zero={int(holdout_zero.sum())} total={int(holdout.sum())}")

    # choose features Z for graph+VGCN
    if args.use_svd_feat:
        print(f"[FEAT] using SVD features for graph+VGCN, k_use={args.k_use}")
        mu = X_in.mean(axis=0)
        Xc = X_in - mu
        svd = TruncatedSVD(n_components=int(args.k_use), random_state=args.seed).fit(Xc)
        Vt_k = svd.components_[:int(args.k_use), :]
        Z = Xc @ Vt_k.T                   # (cells, k)
        recon_back = True
    else:
        print("[FEAT] using raw X features for graph+VGCN (cells x genes). This is heavy but ok for n=2717.")
        Z = X_in.astype(np.float32)
        recon_back = False

    # ====== define save path ======
    graph_dir = "./cache_graph"
    os.makedirs(graph_dir, exist_ok=True)

    graph_name = (
        f"cellgraph_topK{args.topK}_"
        f"shrink{args.shrink_alpha}_"
        f"lam{args.gl_lam}_"
        f"rho{args.gl_rho}_"
        f"iter{args.admm_max_iter}.npz"
    )

    graph_path = os.path.join(graph_dir, graph_name)

    # ====== check if exists ======
    if os.path.exists(graph_path):
        print("Loading cached graph from:", graph_path)
        A = sp.load_npz(graph_path)
    else:
        print("Building graph with ADMM...")
        # build graph (ADMM precision on cell-cell corr)
        A = build_cellcell_graph_admm(
            Z,
            topK=args.topK,
            shrink_alpha=args.shrink_alpha,
            gl_lam=args.gl_lam,
            gl_rho=args.gl_rho,
            admm_max_iter=args.admm_max_iter,
            zero_diag=False
        )
        print("Saving graph to:", graph_path)
        sp.save_npz(graph_path, A)

    # train VGCN to denoise Z
    X_target = X_true
    obs_mask = (~holdout).astype(np.float32)

    t0 = time.time()
    Z_hat, Z_mu, pi0_np = vgcn_denoise_X(
        Z, A,
        X_target=X_target,
        obs_mask=obs_mask,
        holdout_pos_mask=holdout_pos.astype(np.float32),
        holdout_zero_mask=holdout_zero.astype(np.float32),
        epochs=args.epochs,
        lr=args.lr,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        x_drop_prob=args.x_drop_prob,
        seed=args.seed,
        add_self_loop=True
    )
    print(f"[VGCN] done, time={time.time()-t0:.1f}s, Z_hat shape={Z_hat.shape}")
    print("[pi0] holdout_zero mean:", float(pi0_np[holdout_zero].mean()))
    print("[pi0] holdout_pos  mean:", float(pi0_np[holdout_pos].mean()))

    # reconstruct X_hat
    if recon_back:
        # back to X space if using SVD features (nouse for now)
        X_hat = Z_hat @ Vt_k + mu
    else:
        # if Z is X, we already are in X space
        X_hat = Z_hat.astype(np.float64)

    # clip nonneg for safe
    X_hat_clip = np.maximum(X_hat, 0.0)
    neg_frac = float((X_hat < 0).mean())
    print(f"[OUT] X_hat min/max={float(X_hat.min()):.6f}/{float(X_hat.max()):.6f} neg_frac={neg_frac:.6f}")

    # strict eval
    if holdout is not None:
        rmse, mae = rmse_mae(X_true, X_hat_clip, holdout)
        print(f"[EVAL-STRICT clip] RMSE={rmse:.6f}  MAE={mae:.6f}")
    else:
        print("[EVAL-STRICT] skipped (apply_mask=False)")

    # --- detailed holdout diagnostics (pos vs zero) (boolean mask) ---
    if holdout is not None:
        pred = X_hat_clip[holdout]   # use clipped nonneg to match eval
        true = X_true[holdout]

        # NOTE: holdout is sampled from X_true>0, so true==0 should be empty
        zmask = (true == 0)
        pmask = (true > 0)

        if np.any(zmask):
            print("[HOLDOUT true==0] mean(pred) =", float(pred[zmask].mean()),
                  " frac>0.1 =", float((pred[zmask] > 0.1).mean()),
                  " frac>0.2 =", float((pred[zmask] > 0.2).mean()),
                  " frac>0.8 =", float((pred[zmask] > 0.8).mean()))
        else:
            print("[HOLDOUT true==0] skipped (holdout sampled only from X_true>0)")

        if np.any(pmask):
            rmse_p = float(np.sqrt(np.mean((pred[pmask] - true[pmask]) ** 2)))
            mae_p  = float(np.mean(np.abs(pred[pmask] - true[pmask])))
            print("[HOLDOUT true>0] RMSE =", rmse_p, " MAE =", mae_p)


    # save
    import anndata as ad
    X_final = X_true.copy()
    zero_mask = (X_true <= 1e-8)
    pi_thr = 0.3
    val_thr = 0.8
    fill_mask = zero_mask & (pi0_np < pi_thr) & (X_hat_clip > val_thr)
    X_final[fill_mask] = X_hat_clip[fill_mask]
    X_sp = sp.csr_matrix(X_final.astype(np.float32))
    out = ad.AnnData(X=X_sp, obs=adata.obs.copy(), var=adata.var.copy())
    out.obsm["X_vgcn_mu"] = Z_mu.astype(np.float32)
    out.uns["graph_type"] = "cell-cell"
    out.uns["topK"] = int(args.topK)
    out.uns["shrink_alpha"] = float(args.shrink_alpha)
    out.uns["gl_lam"] = float(args.gl_lam)
    out.uns["gl_rho"] = float(args.gl_rho)
    out.uns["use_svd_feat"] = bool(args.use_svd_feat)
    out.uns["k_use"] = int(args.k_use)
    out.uns["seed"] = int(args.seed)
    out.uns["apply_mask"] = bool(args.apply_mask)
    if args.apply_mask:
        out.uns["mask_prob"] = float(args.mask_prob)
        out.uns["mask_seed"] = int(args.mask_seed)
        out.uns["holdout_n"] = int(holdout.sum())
        out.uns["holdout_frac"] = float(holdout.mean())

    tag_feat = f"svd{args.k_use}" if args.use_svd_feat else "rawX"
    tag_mask = f"_mask{args.mask_prob}_ms{args.mask_seed}" if args.apply_mask else ""
    input_tag = os.path.splitext(os.path.basename(args.input))[0]
    out_name = f"{input_tag}_gcn_admm_topK{args.topK}_{tag_feat}{tag_mask}.h5ad"
    out_path = os.path.join(args.out_dir, out_name)
    out.write(out_path)
    print("Saved:", os.path.abspath(out_path))


if __name__ == "__main__":
    main()
