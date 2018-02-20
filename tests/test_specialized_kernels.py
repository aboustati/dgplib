import gpflow
import unittest

import numpy as np
import tensorflow as tf

from dgplib.specialized_kernels import SwitchedKernel

from gpflow.decors import params_as_tensors, defer_build
from gpflow.kernels import Kernel, RBF
from gpflow.params import Parameter
from gpflow.test_util import GPflowTestCase

class DummyKernel(Kernel):
    def __init__(self, input_dim, val, active_dims=None, name=None):
        super(DummyKernel, self).__init__(input_dim, active_dims, name)
        self.val = Parameter(val, dtype=np.float32)

    @params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.val * tf.ones((tf.shape(X)[0], tf.shape(X2)[0]))

    @params_as_tensors
    def Kdiag(self, X):
        return self.val * tf.ones((tf.shape(X)[0], 1))


class SwitchedKernelTest(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()
        self.rng = np.random.RandomState(42)

        self.X1_ind = np.array([0,1,2,2,1,0,1,0,2,1])[:,None]
        self.X1 = np.hstack([np.random.randn(10, 3),
                             self.X1_ind]).astype(gpflow.settings.float_type)

        self.X2_ind = np.array([0,1,2,2,1])[:,None]
        self.X2 = np.hstack([np.random.randn(5, 3),
                             self.X2_ind]).astype(gpflow.settings.float_type)

        with defer_build():
            K1 = DummyKernel(3, 1.0)
            K2 = DummyKernel(3, 2.0)
            K3 = DummyKernel(3, 3.0)
            kern_list = [K1, K2, K3]
            self.kernel = SwitchedKernel(kern_list, 3)

    def test_K(self):
        reference = np.array([[1, 0, 0, 0, 0],
                             [0, 2, 0, 0, 2],
                             [0, 0, 3, 3, 0],
                             [0, 0, 3, 3, 0],
                             [0, 2, 0, 0, 2],
                             [1, 0, 0, 0, 0],
                             [0, 2, 0, 0, 2],
                             [1, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0],
                             [0, 2, 0, 0, 2]])

        with self.test_context() as session:
            X1 = tf.placeholder(gpflow.settings.float_type)
            X2 = tf.placeholder(gpflow.settings.float_type)

            self.kernel.compile()
            gram_matrix = session.run(self.kernel.K(X1, X2), feed_dict={X1:self.X1,
                                                                   X2:self.X2})
            self.assertTrue(np.allclose(gram_matrix, reference))


    def test_Kdiag(self):
        with self.test_context() as session:
            X1 = tf.placeholder(gpflow.settings.float_type)

            self.kernel.compile()
            gram_matrix = session.run(self.kernel.Kdiag(X1), feed_dict={X1:self.X1})
            self.assertTrue(np.allclose(gram_matrix, self.X1[:,-1:]+1))

if __name__=='__main__':
    unittest.main()
