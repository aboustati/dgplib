import tensorflow as tf

from gpflow.kernels import Combination


class SwitchedKernel(Combination):
    """
    This calculates different kernels based on the index in the last column of
    X.
    """
    def __init__(self, kernels, output_dim, name=None):
        super().__init__(kernels=kernels, name=name)
        self.output_dim = output_dim
        self.num_kernels = len(self.kernels)
        assert self.output_dim == self.num_kernels

    def K(self, X, Y=None, presliced=False):
        if Y is None:
            Y = X

        idx_X = X[:, -1]
        idx_X = tf.cast(idx_X, tf.int32)

        idx_Y = Y[:, -1]
        idx_Y = tf.cast(idx_Y, tf.int32)

        X = X[:, :-1]
        Y = Y[:, :-1]

        if not presliced:
            X, Y = self.slice(X, Y)

        idx_X_parts = tf.dynamic_partition(tf.range(0, tf.shape(idx_X)[0]), idx_X, self.output_dim)
        idx_Y_parts = tf.dynamic_partition(tf.range(0, tf.shape(idx_Y)[0]), idx_Y, self.output_dim)

        Ks = []
        for k, p, p2 in zip(self.kernels, idx_X_parts, idx_Y_parts):
            X_gathered = tf.gather(X, p, axis=0, batch_dims=None)
            Y_gathered = tf.gather(Y, p2, axis=0, batch_dims=None)

            gram = k.K(X_gathered, Y_gathered)
            Ks.append(gram)

        N = tf.shape(X)[0]
        N2 = tf.shape(Y)[0]
        Ks_scattered = []
        for gram, p, p2 in zip(Ks, idx_X_parts, idx_Y_parts):
            p2 = p2[:,None]
            shape = tf.stack([N2, tf.shape(p)[0]])
            scattered = tf.transpose(tf.scatter_nd(p2, tf.transpose(gram), shape))
            Ks_scattered.append(scattered)

        return tf.dynamic_stitch(idx_X_parts, Ks_scattered)

    def K_diag(self, X, presliced=False):
        idx_X = X[:, -1]
        idx_X = tf.cast(idx_X, tf.int32)
        X = X[:, :-1]

        ind_X_parts = tf.dynamic_partition(tf.range(0, tf.size(idx_X)), idx_X, self.output_dim)

        Ks = [k.K_diag(tf.gather(X, p, axis=0)) for k, p in zip(self.kernels, ind_X_parts)]

        return tf.dynamic_stitch(ind_X_parts, Ks)
