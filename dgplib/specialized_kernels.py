from __future__ import absolute_import

import tensorflow as tf

from gpflow import settings

from gpflow.kernels import Combination
from gpflow.params import Parameter, Paramlist

class SwitchedKernel(Combination):
    """
    This calculates different kernels based on the index in the last column of
    X.
    """
    def __init__(self, kern_list, output_dim, name=None):
        super(SwitchedKernel, self).__init__(kern_list=kern_list,
                                             name=name)
        self.output_dim = output_dim
        self.num_kernels = len(self.kern_list)
        assert self.output_dim==self.num_kernels

    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            X2 = X

        ind_X= X[:,-1]
        ind_X = tf.cast(ind_X, tf.int32)

        ind_X2= X2[:,-1]
        ind_X2 = tf.cast(ind_X2, tf.int32)

        X = X[:,:-1]
        X2 = X2[:,:-1]

        if prescliced:
            raise NotImplementedError()

        ind_X_parts = tf.dynamic_partition(tf.range(0, tf.size(ind_X)),
                                           ind, self.output_dim)
        ind_X2_parts = tf.dynamic_partition(tf.range(0, tf.size(ind_X2)),
                                           ind, self.output_dim)
        Ks = [k.K(X[p,:], X2[p2,:]) for k, p, p2 in
              zip(self.kern_list, ind_X_parts, ind_X2_parts)]

        N = tf.shape(X)[0]
        N2 = tf.shape(X2)[0]
        Ks_scattered = []
        for k, p2 in zip(Ks, ind_X2_parts):
            p2 = p2[:,None]
            scattered = tf.transpose(tf.scatter(p2, tf.transpose(k),
                                                tf.constant(N2, N)))
            Ks_scattered.append(scattered)

        return tf.dynamic_stitch(ind_X_parts, Ks_scattered)

    def Kdiag(self, X, prescliced=False):
        ind_X= X[:,-1]
        ind_X = tf.cast(ind_X, tf.int32)
        X = X[:,:-1]

        ind_X_parts = tf.dynamic_partition(tf.range(0, tf.size(ind_X)),
                                           ind, self.output_dim)

        Ks = [k.Kdiag(X[p,:]) for k, p in zip(self.kern_list, ind_X_parts)]
        return tf.dynamic_stitch(ind_X_parts, Ks)
