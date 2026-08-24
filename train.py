# --------------------------
# VGCN training: denoise features then reconstruct X
# --------------------------
from layers import scipy_csr_to_tf_sparse, normalize_adj_tf_sparse
from network import VGCN_TPGG
from loss import loss_vgcn_dae_tpgg
import tensorflow as tf
import numpy as np

def vgcn_denoise_X(
    X,
    A,
    X_target=None,  # if None, defaults to X_input_base
    obs_mask=None,
    holdout_pos_mask=None,
    holdout_zero_mask=None,
    epochs=200,
    lr=1e-3,
    hidden1=256,
    hidden2=128,
    latent_dim=64,
    dropout=0.1,
    x_drop_prob=0.2,
    alpha_miss=1.0,
    seed=0,
    add_self_loop=True,
    return_latent=True
):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # --- adjacency ---
    if add_self_loop:
        A_loop = A.copy()
        A_loop.setdiag(1.0)
    else:
        A_loop = A

    A_tf = scipy_csr_to_tf_sparse(A_loop)
    A_norm = normalize_adj_tf_sparse(A_tf)

    # --- X tensor ---
    X_in_tf = tf.convert_to_tensor(X.astype(np.float32))  # input base

    if X_target is None:
        X_target = X
    X_tgt_tf = tf.convert_to_tensor(X_target.astype(np.float32))  # likelihood target

    obs_mask_tf = None
    if obs_mask is not None:
        obs_mask_tf = tf.convert_to_tensor(obs_mask.astype(np.float32))
    holdout_pos_tf = None
    holdout_zero_tf = None
    if holdout_pos_mask is not None:
        holdout_pos_tf = tf.convert_to_tensor(holdout_pos_mask.astype(np.float32))
    if holdout_zero_mask is not None:
        holdout_zero_tf = tf.convert_to_tensor(holdout_zero_mask.astype(np.float32))

    # --- pi0 bias init: per-gene + per-cell ---
    gene_pi_bias_init = None
    cell_pi_bias_init = None

    if obs_mask is not None:
        # obs_mask: 1 means included in loss (NOT strict holdout)
        obs = obs_mask.astype(bool)   # shape (N, G)
        xobs = X_target               # numpy (N, G)

        # per-gene observed zero rate
        gene_zero_rate = ((xobs <= 1e-8) & obs).sum(axis=0) / (obs.sum(axis=0) + 1e-8)  # (G,)

        # per-cell observed zero rate
        cell_zero_rate = ((xobs <= 1e-8) & obs).sum(axis=1) / (obs.sum(axis=1) + 1e-8)  # (N,)
    else:
        xobs = X_target
        gene_zero_rate = (xobs <= 1e-8).mean(axis=0)  # (G,)
        cell_zero_rate = (xobs <= 1e-8).mean(axis=1)  # (N,)

    # clip for numerical stability
    gene_zero_rate = np.clip(gene_zero_rate, 1e-4, 1.0 - 1e-4)
    cell_zero_rate = np.clip(cell_zero_rate, 1e-4, 1.0 - 1e-4)

    # convert zero rates to logits
    gene_pi_bias_init = np.log(gene_zero_rate / (1.0 - gene_zero_rate)).astype(np.float32)

    # cell bias: make it relative, not an extra global positive shift
    cell_pi_bias_raw = np.log(cell_zero_rate / (1.0 - cell_zero_rate)).astype(np.float32)
    cell_pi_bias_raw = cell_pi_bias_raw - cell_pi_bias_raw.mean()

    # optional scaling so the prior is not too strong at start
    gene_bias_scale = 0.5
    cell_bias_scale = 0.3   # smaller than gene bias is safer

    gene_pi_bias_init = gene_bias_scale * gene_pi_bias_init
    cell_pi_bias_init = cell_bias_scale * cell_pi_bias_raw
    print("[init] gene_zero_rate mean=", float(gene_zero_rate.mean()),
          " gene_pi_bias_init mean=", float(gene_pi_bias_init.mean()))
    print("[init] cell_zero_rate mean=", float(cell_zero_rate.mean()),
          " cell_pi_bias_init mean=", float(cell_pi_bias_init.mean()))

    # --- model ---
    model = VGCN_TPGG(
        adj_norm=A_norm,
        n_genes=X.shape[1],
        n_cells=X.shape[0],  # NEW
        hidden1=int(hidden1),
        hidden2=int(hidden2),
        latent_dim=int(latent_dim),
        dropout=float(dropout),
        pi_bias_init=gene_pi_bias_init,  # per-gene
        cell_pi_bias_init=cell_pi_bias_init,  # NEW per-cell
        cell_pi_bias_trainable=True,  # can set False if you want fixed prior
    )
    opt = tf.keras.optimizers.Adam(float(lr))

    # --- constants (avoid retracing) ---
    drop_prob_t  = tf.constant(float(x_drop_prob), tf.float32)
    alpha_miss_t = tf.constant(float(alpha_miss), tf.float32)
    free_bits_t  = tf.constant(0.5, tf.float32)   # 可调：0.0/0.1/0.5

    for ep in range(1, int(epochs) + 1):
        # -------- gamma warmup for BCE --------
        if ep <= 60 :
            gamma_pi_t = tf.constant(0.0, tf.float32)  # stage 1: learn recon first
        else:
            gamma_max = 0.05
            gamma_val = gamma_max * min(1.0, (ep - 60) / 140.0)  # stage 2: ramp
            gamma_pi_t = tf.constant(gamma_val, tf.float32)

        # NOTE: now we pass BOTH target and input base, plus obs_mask
        loss, recon, recon_all, recon_miss, pi_disc = loss_vgcn_dae_tpgg(
            model, opt,
            X_tgt_tf,
            X_in_tf,
            drop_prob_t, alpha_miss_t, gamma_pi_t,
            epoch_t=tf.constant(ep, tf.int32),  # NEW
            free_bits_t=free_bits_t,
            obs_mask=obs_mask_tf,
            holdout_pos_mask=holdout_pos_tf,
            holdout_zero_mask=holdout_zero_tf,
        )

        if ep % 20 == 0:
            print(f"[VGCN-TPGG] ep {ep:4d} loss={float(loss):.6f} recon={float(recon):.6f} "
                  f"all={float(recon_all):.6f} miss={float(recon_miss):.6f} pi_disc={float(pi_disc):.6f}")

    # --- inference ---
    # 先拿 soft output（不做 hard gate）
    mu_x_soft, pi0, a, d, p, mu_gg, mu_z = model(X_in_tf, training=False, tau=None)

    pi = pi0.numpy()
    mu_soft = mu_x_soft.numpy()
    mu_gg_np = mu_gg.numpy()
    Z_mu = mu_z.numpy()

    xobs = X_target
    zero = (xobs <= 1e-8)
    pos = (xobs > 1e-8)

    print("\n[Soft output]")
    print("[pi0] mean on zeros:", pi[zero].mean(), " mean on pos:", pi[pos].mean())
    print("[mu_soft] mean on zeros:", mu_soft[zero].mean(), " mean on pos:", mu_soft[pos].mean(), " max:",
          mu_soft.max())
    print("[mu_gg] mean on zeros:", mu_gg_np[zero].mean(), " mean on pos:", mu_gg_np[pos].mean(), " max:",
          mu_gg_np.max())

    taus = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    def eval_tau(X_hat, X_true, holdout_pos, holdout_zero):
        # pos MAE
        p_pred = X_hat[holdout_pos]
        p_true = X_true[holdout_pos]
        mae_pos = float(np.mean(np.abs(p_pred - p_true)))

        # zero leakage
        z_pred = X_hat[holdout_zero]
        zero_mean = float(z_pred.mean())
        zero_gt01 = float((z_pred > 0.1).mean())
        zero_gt02 = float((z_pred > 0.2).mean())
        zero_gt08 = float((z_pred > 0.8).mean())
        return mae_pos, zero_mean, zero_gt01, zero_gt02, zero_gt08

    best = None
    if holdout_pos_mask is not None and holdout_zero_mask is not None:
        for tau in taus:
            mu_x, pi0, a, d, p, mu_gg, mu_z = model(X_in_tf, training=False, tau=tau)
            X_hat = np.maximum(mu_x.numpy(), 0.0)

            mae_pos, zero_mean, z01, z02, z08 = eval_tau(
                X_hat,
                X_target,
                holdout_pos_mask.astype(bool),
                holdout_zero_mask.astype(bool)
            )
            print(
                f"tau={tau:>4}  MAE_pos={mae_pos:.4f}  zero_mean={zero_mean:.4f}  >0.1={z01:.3f}  >0.2={z02:.3f}  >0.8={z08:.3f}"
            )

            score = mae_pos + 0.5 * zero_mean
            if best is None or score < best["score"]:
                best = {
                    "tau": tau,
                    "score": score,
                    "mae_pos": mae_pos,
                    "zero_mean": zero_mean,
                    "X_hat": X_hat
                }
        X_hat = best["X_hat"]
    else:
        mu_x, pi0, a, d, p, mu_gg, mu_z = model(X_in_tf, training=False, tau=0.5)
        X_hat = np.maximum(mu_x.numpy(), 0.0)

    # 默认返回一个tau
    best_tau = best["tau"]
    print("Best by score:", {k: v for k, v in best.items() if k != "X_hat"})
    if return_latent:
        return X_hat, Z_mu, pi
    return X_hat, pi
