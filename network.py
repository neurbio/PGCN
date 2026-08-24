import numpy as np
import tensorflow as tf
from layers import GraphConvolution

def scipy_csr_to_tf_sparse(A_csr):
    A_csr = A_csr.tocoo()
    indices = np.vstack((A_csr.row, A_csr.col)).T.astype(np.int64)
    values = A_csr.data.astype(np.float32)
    shape = np.array(A_csr.shape, dtype=np.int64)
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

def normalize_adj_tf_sparse(A):
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}
    A: tf.sparse.SparseTensor
    """
    A = tf.sparse.reorder(A)
    deg = tf.sparse.reduce_sum(A, axis=1)  # (N,)
    deg_inv_sqrt = tf.pow(deg + 1e-12, -0.5)

    row = A.indices[:, 0]
    col = A.indices[:, 1]
    new_vals = A.values * tf.gather(deg_inv_sqrt, row) * tf.gather(deg_inv_sqrt, col)

    return tf.sparse.SparseTensor(indices=A.indices, values=new_vals, dense_shape=A.dense_shape)

def add_self_loops_tf_sparse(A):
    """A <- A + I (sparse)."""
    n = int(A.dense_shape[0])
    eye = tf.sparse.SparseTensor(
        indices=tf.cast(tf.stack([tf.range(n), tf.range(n)], axis=1), tf.int64),
        values=tf.ones([n], tf.float32),
        dense_shape=[n, n]
    )
    A = tf.sparse.add(tf.sparse.reorder(A), eye)
    return tf.sparse.reorder(A)

class VGCN_TPGG(tf.keras.Model):
    """
    Variational GCN with TPGG decoder
    """

    def __init__(self, adj_norm, n_genes, n_cells,
                 hidden1=256, hidden2=128, latent_dim=64, dropout=0.0,
                 pi_bias_init=None,           # per-gene bias init
                 cell_pi_bias_init=None,      # NEW: per-cell bias init
                 cell_pi_bias_trainable=True  # NEW: can choose fixed/trainable
                 ):
        super().__init__()

        self.n_genes = int(n_genes)
        self.n_cells = int(n_cells)

        # ---------- Encoder ----------
        self.enc_gcn1 = GraphConvolution(n_genes, hidden1, adj_norm,
                                         activation=tf.nn.relu, dropout=dropout)

        self.enc_mu = GraphConvolution(hidden1, latent_dim, adj_norm)

        # ---------- Decoder MLP ----------
        self.dec_dense1 = tf.keras.layers.Dense(hidden2, activation=tf.nn.relu)
        self.dec_dense2 = tf.keras.layers.Dense(hidden1, activation=tf.nn.relu)

        # ---------- pi head ----------
        # Keep per-gene bias inside Dense (same as your current design)
        if pi_bias_init is None:
            self.dec_logit_pi = tf.keras.layers.Dense(n_genes)
        else:
            self.dec_logit_pi = tf.keras.layers.Dense(
                n_genes,
                bias_initializer=tf.keras.initializers.Constant(pi_bias_init),
            )

        # NEW: per-cell bias added on top of gene-wise bias
        if cell_pi_bias_init is None:
            cell_pi_bias_init = np.zeros((self.n_cells,), dtype=np.float32)

        self.cell_pi_bias = self.add_weight(
            name="cell_pi_bias",
            shape=(self.n_cells,),
            initializer=tf.keras.initializers.Constant(cell_pi_bias_init),
            trainable=bool(cell_pi_bias_trainable),
        )

        # ---------- TPGG parameters ----------
        self.dec_eta_a = tf.keras.layers.Dense(n_genes)
        self.dec_eta_d = tf.keras.layers.Dense(n_genes)
        self.dec_eta_p = tf.keras.layers.Dense(n_genes)

    # ---------- Encoder ----------
    def encode(self, X, training=False):
        h1 = self.enc_gcn1(X, training=training)
        mu = self.enc_mu(h1, training=training)
        return mu

    def reparameterize(self, mu, logvar, training=False):
        if training:
            eps = tf.random.normal(tf.shape(mu))
            return mu + eps * tf.exp(0.5 * logvar)
        return mu

    # ---------- GG mean ----------
    def gg_mean(self, a, d, p):
        return a * tf.exp(
            tf.math.lgamma(d + 1.0 / p) - tf.math.lgamma(d)
        )

    # ---------- Decoder ----------
    def decode(self, z, training=False):
        y = self.dec_dense1(z)
        y = self.dec_dense2(y)

        # base logit from decoder (includes per-gene bias via Dense bias)
        logit_pi = self.dec_logit_pi(y)   # shape (N, G)

        # NEW: add per-cell bias (broadcast from (N,) -> (N, 1))
        logit_pi = logit_pi + self.cell_pi_bias[:, tf.newaxis]

        # dropout probability
        pi0 = tf.sigmoid(logit_pi)

        # GG params > 0
        a = tf.nn.softplus(self.dec_eta_a(y)) + 1e-8
        d = tf.nn.softplus(self.dec_eta_d(y)) + 1e-8
        p = tf.nn.softplus(self.dec_eta_p(y)) + 1e-8

        mu_gg = self.gg_mean(a, d, p)

        return pi0, a, d, p, mu_gg

    # ---------- Forward ----------
    def call(self, X, training=False, tau=None, scale_c=0.5):
        mu_z = self.encode(X, training=training)

        pi0, a, d, p, mu_gg = self.decode(mu_z, training=training)

        mu_x_soft = (1.0 - pi0) * mu_gg * scale_c
        mu_x = mu_x_soft

        if (not training) and (tau is not None):
            mu_x = tf.where(pi0 > tau, tf.zeros_like(mu_gg), mu_gg * scale_c)

        return mu_x, pi0, a, d, p, mu_gg, mu_z