import numpy as np
import tensorflow as tf

def scipy_csr_to_tf_sparse(A_csr):
    A_csr = A_csr.tocoo()
    indices = np.vstack((A_csr.row, A_csr.col)).T.astype(np.int64)
    values = A_csr.data.astype(np.float32)
    shape = np.array(A_csr.shape, dtype=np.int64)
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

def normalize_adj_tf_sparse(A):
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}
    A must be tf.sparse.SparseTensor (assumed nonnegative weights).
    """
    A = tf.sparse.reorder(A)
    # degree: sum over rows
    deg = tf.sparse.reduce_sum(A, axis=1)  # (N,)
    deg_inv_sqrt = tf.pow(deg + 1e-12, -0.5)
    # build D^{-1/2} * A * D^{-1/2} via value scaling
    row = A.indices[:, 0]
    col = A.indices[:, 1]
    new_vals = A.values * tf.gather(deg_inv_sqrt, row) * tf.gather(deg_inv_sqrt, col)
    return tf.sparse.SparseTensor(indices=A.indices, values=new_vals, dense_shape=A.dense_shape)

class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, adj, activation=None, dropout=0.0, use_bias=True, name=None):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj = adj  # tf.sparse.SparseTensor, normalized ideally
        self.activation = activation
        self.dropout = dropout
        self.use_bias = use_bias

        self.W = self.add_weight(
            shape=(input_dim, output_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="W"
        )
        if use_bias:
            self.b = self.add_weight(
                shape=(output_dim,),
                initializer="zeros",
                trainable=True,
                name="b"
            )
        else:
            self.b = None

    def call(self, x, training=False):
        # x: (N, input_dim)
        if training and self.dropout and self.dropout > 0:
            x = tf.nn.dropout(x, rate=self.dropout)

        xW = tf.matmul(x, self.W)  # (N, output_dim)
        AxW = tf.sparse.sparse_dense_matmul(self.adj, xW)  # (N, output_dim)

        if self.use_bias:
            AxW = AxW + self.b

        if self.activation is not None:
            AxW = self.activation(AxW)
        return AxW

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, activation=tf.nn.relu, dropout=0.0, name=None):
        super().__init__(name=name)
        self.dropout = dropout
        self.dense = tf.keras.layers.Dense(output_dim, activation=None)
        self.activation = activation

    def call(self, x, training=False):
        if training and self.dropout and self.dropout > 0:
            x = tf.nn.dropout(x, rate=self.dropout)
        x = self.dense(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
