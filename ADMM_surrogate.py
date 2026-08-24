import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def soft_threshold(M, tau, penalize_diag=False):
    X = np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)
    if not penalize_diag:
        np.fill_diagonal(X, np.diag(M))
    return X

def admm_sparse_precision(
    S,
    lam=0.1,
    rho=1e-2,
    max_iter=10,
    reltol=1e-4,
    abstol=1e-5,
    penalize_diag=False,
    zero_diag=True,
    return_history=False,
    mask=None,
    seed=None
):
    # --- sanitize / stability ---
    S = np.asarray(S, dtype=np.float64, order="C")
    S = 0.5 * (S + S.T)
    if not np.isfinite(S).all():
        raise ValueError("S contains NaN/Inf")

    p = S.shape[0]
    I = np.eye(p, dtype=np.float64)

    # A = rho I + S (SPD if rho>0 and S PSD). add tiny ridge for numerical safety
    ridge = 1e-6
    A = S + (rho + ridge) * I

    # Cholesky factorization once
    c, lower = cho_factor(A, check_finite=False)

    Theta = I.copy()
    Phi   = I.copy()
    Psi   = np.zeros((p, p), dtype=np.float64)
    Phi_prev = Phi.copy()

    hist = {"r_norm": [], "s_norm": [], "eps_pri": [], "eps_dual": []}
    tau = rho / lam  # keep your convention

    for k in range(1, max_iter + 1):
        # ---- Θ-update: Theta_bar = (rho I + S)^(-1) * (I + rho(Phi - Psi)) ----
        RHS = I + rho * (Phi - Psi)
        Theta_bar = cho_solve((c, lower), RHS, check_finite=False)
        Theta = 0.5 * (Theta_bar + Theta_bar.T)

        # ---- Φ-update ----
        V = Theta + Psi
        Phi = soft_threshold(V, tau, penalize_diag=penalize_diag)
        Phi = 0.5 * (Phi + Phi.T)

        if mask is not None:
            # mask: bool or 0/1 float, same shape
            Phi *= mask
            if not penalize_diag:
                np.fill_diagonal(Phi, np.diag(V))

        # ---- Ψ-update ----
        Psi = Psi + (Theta - Phi)

        # ---- stopping ----
        r = Theta - Phi
        s = rho * (Phi - Phi_prev)

        r_norm = np.linalg.norm(r, 'fro')
        s_norm = np.linalg.norm(s, 'fro')

        eps_pri = np.sqrt(p*p) * abstol + reltol * max(
            np.linalg.norm(Theta, 'fro'),
            np.linalg.norm(Phi,   'fro')
        )
        eps_dual = np.sqrt(p*p) * abstol + reltol * rho * np.linalg.norm(Psi, 'fro')

        if return_history:
            hist["r_norm"].append(r_norm)
            hist["s_norm"].append(s_norm)
            hist["eps_pri"].append(eps_pri)
            hist["eps_dual"].append(eps_dual)

        if (r_norm <= eps_pri) and (s_norm <= eps_dual):
            break

        Phi_prev = Phi.copy()

    Theta = 0.5 * (Theta + Theta.T)

    if zero_diag:
        Aout = Theta.copy()
        np.fill_diagonal(Aout, 0.0)
    else:
        Aout = Theta

    return (Theta, Aout, hist) if return_history else (Theta, Aout)

if __name__ == "__main__":
    dis = np.load('./data/S.npy')
    print("type(S):", type(dis))
    try:
        import scipy.sparse as sp

        print("is sparse:", sp.issparse(dis))
    except Exception as e:
        print("sparse check error:", e)
    print("ndim:", np.ndim(dis))
    print("shape:", getattr(dis, "shape", None))

    for thr in [0.05, 0.1, 0.2, 0.3]:
        nnz = np.count_nonzero(np.abs(dis) >= thr)
        density = nnz / dis.size
        print(f"thr={thr}: nnz={nnz:,}, density={density:.6f}, sparsity={1 - density:.6f}")

    '''
    k = 50
    abs_dis = np.abs(dis)
    np.fill_diagonal(abs_dis, 0.0)

    idx = np.argpartition(-abs_dis, kth=k-1, axis=1)[:, :k]
    rows = np.repeat(np.arange(dis.shape[0]), k)
    cols = idx.reshape(-1)
    vals = dis[rows, cols]

    Asp = sp.csr_matrix((vals, (rows, cols)), shape=dis.shape)
    Asp = 0.5 * (Asp + Asp.T)
    '''

    thr = 0.2
    C = dis.copy()
    np.fill_diagonal(C, 0.0)
    C[np.abs(C) < thr] = 0.0
    G0 = sp.csr_matrix(C)
    G0.eliminate_zeros()
    print(G0.shape, G0.nnz)

    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components


    def topk_graph_from_corr(dis, k=10, sym=True):
        dis = dis.astype(np.float32, copy=False)
        p = dis.shape[0]

        A = dis.copy()
        np.fill_diagonal(A, 0.0)
        absA = np.abs(A)

        idx = np.argpartition(-absA, kth=k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(p), k)
        cols = idx.reshape(-1)
        vals = A[rows, cols]

        G = sp.csr_matrix((vals, (rows, cols)), shape=(p, p))
        if sym:
            G = 0.5 * (G + G.T)
        G.eliminate_zeros()
        return G

    def mutual_topk_graph_from_corr(dis, k=10):
        dis = dis.astype(np.float32, copy=False)
        p = dis.shape[0]

        A = dis.copy()
        np.fill_diagonal(A, 0.0)
        absA = np.abs(A)

        idx = np.argpartition(-absA, kth=k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(p), k)
        cols = idx.reshape(-1)
        vals = A[rows, cols]

        # directed kNN
        Gdir = sp.csr_matrix((np.ones_like(vals, dtype=np.int8), (rows, cols)), shape=(p, p))
        # mutual edges: i->j and j->i
        Gmut = Gdir.multiply(Gdir.T)  # keeps only reciprocal
        # put weights back (optional: keep original corr weights)
        Gw = sp.csr_matrix((vals, (rows, cols)), shape=(p, p))
        G = Gmut.multiply(Gw)  # keep weights only on mutual edges
        G = 0.5 * (G + G.T)
        G.eliminate_zeros()
        return G

    for k in [3, 5, 8, 10, 15, 20]:
        G0 = mutual_topk_graph_from_corr(dis, k=k)
        n_comp, labels = connected_components(G0, directed=False)
        print(f"k={k}: components={n_comp}, nnz={G0.nnz}")



    # G0: csr_matrix (gene-gene graph)
    n_comp, labels = connected_components(G0, directed=False, connection='weak')
    print("components:", n_comp)

    # labels[i] = component id of gene i
    components = {}
    for i, cid in enumerate(labels):
        components.setdefault(cid, []).append(i)

    # 转成 list，并按大小排序
    comp_list = sorted(components.values(), key=len, reverse=True)

    # 看规模分布
    sizes = [len(c) for c in comp_list]
    print("largest component:", sizes[0])
    print("top 10 sizes:", sizes[:10])

    import numpy as np

    # G0: csr_matrix (p,p) from thresholded correlation
    n_comp, labels = connected_components(G0, directed=False, connection='weak')

    # 找最大分量 id
    comp_sizes = np.bincount(labels)
    largest_cid = comp_sizes.argmax()
    idx_big = np.where(labels == largest_cid)[0]
    print("largest component size:", idx_big.size)  # 5750

    import scipy.sparse as sp

    # ---------- Inputs ----------
    S_full = dis.astype(np.float32, copy=False)  # dense correlation
    G0 = G0.tocsr()  # sparse candidate graph (thr=0.2)

    max_block = 6000
    min_block = 10  # tiny blocks: skip ADMM, keep PCC/topk later
    lam = 0.2
    rho = 0.05
    theta_thr = 1e-3


    # ---------- helper: split a set of nodes by raising threshold inside that set ----------
    def split_inside_block(global_idx, S_full, thr_list=(0.20, 0.30, 0.35, 0.40)):
        """
        global_idx: 1D array of global gene indices
        Returns: list of sub-blocks (each is global indices)
        """
        idx = np.array(global_idx, dtype=np.int32)
        Cb = S_full[np.ix_(idx, idx)].astype(np.float32, copy=False)
        np.fill_diagonal(Cb, 0.0)

        for thr2 in thr_list:
            Ab = Cb.copy()
            Ab[np.abs(Ab) < thr2] = 0.0
            Gb = sp.csr_matrix(Ab)
            Gb.eliminate_zeros()
            ncomp, lab = connected_components(Gb, directed=False, connection="weak")
            parts = []
            for cid in range(ncomp):
                loc = np.where(lab == cid)[0]
                if loc.size > 0:
                    parts.append(idx[loc])  # map back to global
            parts.sort(key=len, reverse=True)
            if len(parts) == 0:
                continue
            if len(parts[0]) <= max_block:
                print(f"[split] thr2={thr2} ok, largest={len(parts[0])}, nblocks={len(parts)}")
                return parts

        # if still too big, fall back to simple chunking (last resort)
        print("[split] threshold split not enough; falling back to chunking")
        return [idx[i:i + max_block] for i in range(0, len(idx), max_block)]


    # ---------- build component list ----------
    n_comp, labels = connected_components(G0, directed=False, connection="weak")
    components = {}
    for i, cid in enumerate(labels):
        components.setdefault(cid, []).append(i)
    comp_list = sorted(components.values(), key=len, reverse=True)

    print("components:", n_comp, "largest:", len(comp_list[0]))

    # ---------- blocks to run ADMM on ----------
    blocks = []

    # small components (skip the biggest for now)
    for comp in comp_list[1:]:
        if len(comp) >= min_block:
            blocks.append(np.array(comp, dtype=np.int32))

    # split the biggest component into <= max_block sub-blocks
    big_block = np.array(comp_list[0], dtype=np.int32)
    blocks.extend(split_inside_block(big_block, S_full))

    print("total blocks for ADMM:", len(blocks), "max block size:", max(len(b) for b in blocks))

    blocks = [b for b in blocks if len(b) >= 30]
    print("blocks after min size:", len(blocks), "max size:", max(len(b) for b in blocks))

    # ---------- run masked ADMM per block and assemble sparse Theta ----------
    rows, cols, vals = [], [], []

    for bi, idx in enumerate(blocks, 1):
        m = len(idx)
        if m < min_block:
            continue
        if m > max_block:
            print("skip too-large block (should not happen):", m)
            continue

        # S_sub: dense correlation in block
        S_sub = S_full[np.ix_(idx, idx)].astype(np.float32, copy=False)
        S_sub = 0.5 * (S_sub + S_sub.T)
        S_sub = S_sub + 1e-4 * np.eye(m, dtype=np.float32)

        # mask_sub from G0: allow edges that exist in G0 OR diagonal
        M = G0[np.ix_(idx, idx)].toarray()
        mask_sub = (M != 0)
        np.fill_diagonal(mask_sub, 1.0)
        print("mask density:", mask_sub.mean(), "mask nnz:", mask_sub.sum())

        Theta_sub, _ = admm_sparse_precision(
            S_sub,
            lam=lam,
            rho=rho,
            max_iter=300,
            reltol=1e-4,
            abstol=1e-5,
            penalize_diag=False,
            zero_diag=True,
            return_history=False,
            mask=mask_sub
        )

        # sparsify Theta_sub for storage
        np.fill_diagonal(Theta_sub, 0.0)
        ii, jj = np.where(np.abs(Theta_sub) >= theta_thr)
        vv = Theta_sub[ii, jj].astype(np.float32)

        rows.extend(idx[ii].tolist())
        cols.extend(idx[jj].tolist())
        vals.extend(vv.tolist())

        if bi % 10 == 0 or bi == len(blocks):
            print(f"block {bi}/{len(blocks)} done, size={m}, edges_kept={len(vv)}")

    seed_idx = max(blocks, key=len)
    print("seed size:", len(seed_idx))
    # print('seeds',seed_idx)
    seed_set = set(seed_idx.tolist())
    p = dis.shape[0]

    Theta_global = sp.csr_matrix((vals, (rows, cols)), shape=(S_full.shape[0], S_full.shape[0]))
    Theta_global = 0.5 * (Theta_global + Theta_global.T)
    Theta_global.eliminate_zeros()
    deg_theta = np.diff(Theta_global.tocsr().indptr)
    print("min/mean/max degree:", deg_theta.min(), deg_theta.mean(), deg_theta.max())

    def connect_to_seed_balanced(dis, seed_idx, k_min=20, cap_in=800, post_thr=None):
        """
        强制每个非-seed gene 至少连 k_min 个 seed gene，
        同时限制每个 seed gene 被连接的次数（cap_in），避免超级 hub。
        post_thr: 可选，过滤掉 |corr| < post_thr 的补边（例如 0.05）
        """
        p = dis.shape[0]
        seed_idx = np.array(seed_idx, dtype=np.int32)
        s = seed_idx.size

        in_seed = np.zeros(p, dtype=bool)
        in_seed[seed_idx] = True

        seed_in = np.zeros(s, dtype=np.int32)  # incoming counter per seed (local index)

        rows, cols, vals = [], [], []

        for i in range(p):
            if in_seed[i]:
                continue

            c = dis[i, seed_idx]
            abs_c = np.abs(c)

            # try more candidates than needed so we can skip capped seeds
            k_try = min(max(k_min * 4, k_min), s)
            cand = np.argpartition(-abs_c, kth=min(k_try - 1, s - 1))[:k_try]
            cand = cand[np.argsort(-abs_c[cand])]

            chosen = []
            for jloc in cand:
                if seed_in[jloc] < cap_in:
                    chosen.append(jloc)
                    seed_in[jloc] += 1
                    if len(chosen) == k_min:
                        break

            # fallback (rare): scan more if many are capped
            if len(chosen) < k_min:
                cand2 = np.argsort(-abs_c)
                for jloc in cand2:
                    if seed_in[jloc] < cap_in:
                        chosen.append(jloc)
                        seed_in[jloc] += 1
                        if len(chosen) == k_min:
                            break

            if len(chosen) == 0:
                continue

            chosen = np.array(chosen, dtype=np.int32)
            top_global = seed_idx[chosen]

            rows.extend([i] * len(chosen))
            cols.extend(top_global.tolist())
            vals.extend(c[chosen].astype(np.float32).tolist())

        A = sp.csr_matrix((vals, (rows, cols)), shape=(p, p), dtype=np.float32)
        A = 0.5 * (A + A.T)
        A.eliminate_zeros()

        if post_thr is not None:
            A = A.tocsr()
            A.data[np.abs(A.data) < post_thr] = 0.0
            A.eliminate_zeros()
        return A


    A_seed_links = connect_to_seed_balanced(dis, seed_idx, k_min=12, cap_in=400, post_thr=None)
    print("A_seed_links nnz:", A_seed_links.nnz)

    print("Theta_global:", Theta_global.shape, "nnz:", Theta_global.nnz)
    G_final = Theta_global.tocsr().astype(np.float32, copy=False)
    G_final = G_final + A_seed_links
    G_final = 0.5 * (G_final + G_final.T)
    G_final.eliminate_zeros()

    # sp.save_npz("./data/MES10426_graph_for_imputation.npz", G_final, compressed=False)
    print("final nnz:", G_final.nnz)

    deg = np.diff(G_final.tocsr().indptr)
    print("min/mean/max degree:", deg.min(), deg.mean(), deg.max())
    print("percent degree==0:", (deg == 0).mean())
    print("percent degree<5:", (deg < 5).mean())
    print("percent degree<20:", (deg < 20).mean())
