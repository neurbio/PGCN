import tensorflow as tf

def kl_standard_normal(mu, logvar):
    return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1.0 - logvar, axis=1)

def tpgg_nll(x, pi0, a, d, p, eps=1e-8, mask=None):
    """
    TPGG NLL:
      pi0 = P(X=0)  (spike at zero)
      GG used only for x>0
    """
    x = tf.maximum(x, 0.0)
    pi0 = tf.cast(pi0, tf.float32)
    a  = tf.cast(a,  tf.float32)
    d  = tf.cast(d,  tf.float32)
    p  = tf.cast(p,  tf.float32)

    # stability
    pi0 = tf.clip_by_value(pi0, 1e-6, 1.0 - 1e-6)

    is_zero = tf.cast(x <= 1e-9, tf.float32)   # 1 if x==0 else 0

    # GG logpdf computed on positive x only (avoid x=0 affecting gradients)
    x_pos = tf.maximum(x, eps)
    gg_lp_full = gg_log_prob(x_pos, a, d, p, eps=eps)

    # IMPORTANT: remove GG contribution for zero entries
    gg_lp = tf.where(is_zero > 0.0, tf.zeros_like(gg_lp_full), gg_lp_full)

    log_p0  = tf.math.log(pi0)                 # x==0
    log_pnz = tf.math.log1p(-pi0) + gg_lp      # x>0

    logp = is_zero * log_p0 + (1.0 - is_zero) * log_pnz
    nll = -logp

    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        return tf.reduce_sum(nll * mask) / (tf.reduce_sum(mask) + 1e-8)
    return tf.reduce_mean(nll)

EPS = 1e-8

def gg_log_prob(x, a, d, p, eps=1e-8):
    x = tf.cast(x, tf.float32)
    a = tf.cast(a, tf.float32)
    d = tf.cast(d, tf.float32)
    p = tf.cast(p, tf.float32)
    a = tf.maximum(a, 1e-6)
    d = tf.maximum(d, 1e-6)
    p = tf.maximum(p, 1e-6)

    x_pos = tf.maximum(x, eps)
    dp = d * p
    return (
        tf.math.log(p + eps)
        - tf.math.lgamma(d + eps)
        + (dp - 1.0) * tf.math.log(x_pos)
        - dp * tf.math.log(a)
        - tf.pow(x_pos / a, p)
    )

@tf.function
def loss_vgcn_dae_tpgg(
    model, opt,
    X_target, X_input_base,
    drop_prob_t, alpha_miss_t, gamma_pi_t,
    epoch_t,
    free_bits_t=None,
    obs_mask=None,
    holdout_pos_mask=None,
    holdout_zero_mask=None,
):
    # ---- corruption: only corrupt positives ----
    xpos = tf.cast(X_target > 0.0, X_input_base.dtype)
    rnd  = tf.random.uniform(tf.shape(X_input_base), dtype=X_input_base.dtype)
    keep = tf.where(xpos > 0.0,
                    tf.cast(rnd > drop_prob_t, X_input_base.dtype),
                    tf.ones_like(X_input_base, dtype=X_input_base.dtype))
    X_in = X_input_base * keep
    miss = 1.0 - keep

    valid = tf.ones_like(tf.cast(X_target, tf.float32))
    if obs_mask is not None:
        valid = tf.cast(obs_mask, tf.float32)
        miss  = miss * valid

    # ---- GG-first schedule ----
    epoch_t  = tf.cast(epoch_t, tf.int32)
    stage1   = epoch_t <= 30
    stage3   = epoch_t > 100

    lambda_gate    = tf.where(stage1, 0.0, 0.1)
    lambda_pos     = tf.constant(1.0, tf.float32)
    lambda_miss    = tf.where(stage1, 1.0 * alpha_miss_t, alpha_miss_t)
    lambda_cons    = tf.constant(0.1, tf.float32)
    lambda_zero_mu = tf.where(stage1, 0.0, 0.03)
    lambda_couple  = tf.constant(0.02, tf.float32)

    gamma_pi_eff = tf.where(stage3,
                            tf.minimum(gamma_pi_t, tf.constant(0.002, tf.float32)),
                            tf.constant(0.0, tf.float32))

    with tf.GradientTape() as tape:
        # model forward (your model returns 7 items here)
        x_hat, pi0, a, d, p, mu_gg, mu_z = model(X_in, training=True)
        pi0 = tf.clip_by_value(pi0, 1e-6, 1.0 - 1e-6)

        # ---- gate loss (balanced BCE) ----
        zero_label = tf.cast(X_target <= 1e-8, tf.float32)

        zero_frac = tf.reduce_sum(zero_label * valid) / (tf.reduce_sum(valid) + EPS)
        pos_frac  = 1.0 - zero_frac
        w_zero    = 0.5 / (zero_frac + EPS)
        w_pos     = 0.5 / (pos_frac  + EPS)

        gate_bce  = -(w_zero * zero_label * tf.math.log(pi0) +
                      w_pos  * (1.0 - zero_label) * tf.math.log1p(-pi0))
        gate_loss = tf.reduce_sum(gate_bce * valid) / (tf.reduce_sum(valid) + EPS)

        # ---- GG NLL: compute safely, but only supervise on positives ----
        pos_mask = tf.cast(X_target > 1e-8, tf.float32) * valid
        x_pos    = tf.where(X_target > 1e-8, tf.cast(X_target, tf.float32), tf.ones_like(tf.cast(X_target, tf.float32)))
        gg_nll   = -gg_log_prob(x_pos, a, d, p)  # can be negative (density>1), that's OK

        pos_w    = pos_mask * (1.0 + 0.5 * tf.cast(X_target, tf.float32))
        pos_loss = tf.reduce_sum(gg_nll * pos_w) / (tf.reduce_sum(pos_w) + EPS)

        miss_pos = tf.cast(miss, tf.float32) * tf.cast(X_target > 1e-8, tf.float32) * valid
        pos_miss_loss = tf.reduce_sum(gg_nll * miss_pos) / (tf.reduce_sum(miss_pos) + EPS)

        cons_w   = pos_mask * (1.0 + 1.0 * tf.cast(X_target, tf.float32))
        cons_err = tf.square(mu_gg - tf.cast(X_target, tf.float32))
        consistency_loss = tf.reduce_sum(cons_err * cons_w) / (tf.reduce_sum(cons_w) + EPS)

        # ---- zero-side GG suppression (hinge) + stop-grad coupling ----
        zero_mask  = tf.cast(X_target <= 1e-8, tf.float32) * valid

        zero_margin = tf.constant(0.15, tf.float32)
        zero_excess = tf.nn.relu(mu_gg - zero_margin)
        zero_mu_penalty = tf.reduce_sum(tf.square(zero_excess) * zero_mask) / (tf.reduce_sum(zero_mask) + EPS)

        couple_penalty = tf.reduce_sum(
            tf.stop_gradient(pi0) * tf.square(mu_gg) * zero_mask
        ) / (tf.reduce_sum(zero_mask) + EPS)

        # ---- optional strict holdout pi calibration (late + tiny) ----
        pi_disc = tf.constant(0.0, tf.float32)
        if (holdout_pos_mask is not None) and (holdout_zero_mask is not None):
            hp = tf.cast(holdout_pos_mask, tf.float32)
            hz = tf.cast(holdout_zero_mask, tf.float32)
            loss_hz = tf.reduce_sum((-tf.math.log(pi0))      * hz) / (tf.reduce_sum(hz) + EPS)
            loss_hp = tf.reduce_sum((-tf.math.log1p(-pi0))   * hp) / (tf.reduce_sum(hp) + EPS)
            pi_disc = loss_hz + loss_hp

        recon = (lambda_gate    * gate_loss +
                 lambda_pos     * pos_loss +
                 lambda_miss    * pos_miss_loss +
                 lambda_cons    * consistency_loss +
                 lambda_zero_mu * zero_mu_penalty +
                 lambda_couple  * couple_penalty)

        loss = recon + gamma_pi_eff * pi_disc

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads]
    opt.apply_gradients(zip(grads, model.trainable_variables))

    recon_all  = gate_loss + pos_loss
    recon_miss = pos_miss_loss
    return loss, recon, recon_all, recon_miss, pi_disc