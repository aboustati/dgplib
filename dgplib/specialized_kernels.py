from __future__ import absolute_import

import tensorflow as tf

from gpflow import settings

from gpflow.decors import params_as_tensors
from gpflow.kernels import Combination
from gpflow.params import Parameter

class SwitchedKernel(Combination):
    """
    This calculates different kernels based on the index in the last column of
    X.
    """
    def __init__(self, kern_list, output_dim, name=None):
        super(SwitchedKernel, self).__init__(kernels=kern_list,
                                             name=name)
        self.output_dim = output_dim
        self.num_kernels = len(self.kern_list)
        assert self.output_dim==self.num_kernels

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            X2 = X

        ind_X = X[:,-1]
        ind_X = tf.cast(ind_X, tf.int32)

        ind_X2 = X2[:,-1]
        ind_X2 = tf.cast(ind_X2, tf.int32)

        X = X[:,:-1]
        X2 = X2[:,:-1]

        if not presliced:
            X, X2 = self._slice(X, X2)

        ind_X_parts = tf.dynamic_partition(tf.range(0, tf.size(ind_X)),
                                           ind_X, self.output_dim)
        ind_X2_parts = tf.dynamic_partition(tf.range(0, tf.size(ind_X2)),
                                           ind_X2, self.output_dim)

        Ks = []
        for k, p, p2 in zip(self.kern_list, ind_X_parts, ind_X2_parts):
            gram = k.K(tf.gather(X, p), tf.gather(X2, p2))
            Ks.append(gram)

        N = tf.shape(X)[0]
        N2 = tf.shape(X2)[0]
        Ks_scattered = []
        for gram, p, p2 in zip(Ks, ind_X_parts, ind_X2_parts):
            p2 = p2[:,None]
            shape = tf.stack([N2, tf.shape(p)[0]])
            scattered = tf.transpose(tf.scatter_nd(p2, tf.transpose(gram),
                                                   shape))
            Ks_scattered.append(scattered)

        return tf.dynamic_stitch(ind_X_parts, Ks_scattered)

    @params_as_tensors
    def Kdiag(self, X, prescliced=False):
        ind_X = X[:,-1]
        ind_X = tf.cast(ind_X, tf.int32)
        X = X[:,:-1]

        ind_X_parts = tf.dynamic_partition(tf.range(0, tf.size(ind_X)),
                                           ind_X, self.output_dim)

        Ks = [k.Kdiag(tf.gather(X, p)) for k, p in zip(self.kern_list, ind_X_parts)]
        return tf.dynamic_stitch(ind_X_parts, Ks)
