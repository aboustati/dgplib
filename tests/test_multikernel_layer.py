#temporary
#import sys
#sys.path.append('../dgplib/')
#import layers

import unittest

import numpy as np
import tensorflow as tf

from dgplib.layers import find_weights, Layer, InputLayer, HiddenLayer, OutputLayer
from dgplib.multikernel_layers import MultikernelLayer

from gpflow.decors import defer_build, autoflow, params_as_tensors
from gpflow.kernels import White, RBF
from gpflow.mean_functions import Linear
from gpflow.models import Model
from gpflow.params import Parameter
from gpflow.test_util import GPflowTestCase

class MultikernelLayerTest(unittest.TestCase):
    @defer_build()
    def prepare(self):
        rng = np.random.RandomState(42)
        num_output = 3
        num_inducing = 10
        kern_list = [RBF(1) for _ in range(num_output)]
        Z = np.random.randn(num_inducing, 1)
        X = np.random.randn(20, 1)

        layer = MultikernelLayer(1, num_output, num_inducing, kern_list, share_Z=False)
        for z in layer.Z:
            z.assign(Z.copy())

        layer_shared_Z = MultikernelLayer(1, num_output, num_inducing, kern_list, share_Z=True)
        layer_shared_Z.Z.assign(Z)

        return X, Z , layer, layer_shared_Z, kern_list

    @defer_build()
    def prepare_autoflow_functions(self, layer):
        class LayerAsModel(Model):
            def __init__(self, layer):
                super(LayerAsModel, self).__init__()
                self.layer = layer
                self.a = Parameter(3.)

            @params_as_tensors
            def _build_likelihood(self):
                return -tf.square(self.a)

            @autoflow((tf.float64, [None, None]))
            def predict(self, Xnew):
                return self.layer._build_predict(Xnew, stochastic=False)

            @autoflow((tf.float64, [None, None]))
            def predict_full_cov(self, Xnew):
                return self.layer._build_predict(Xnew, full_cov=True, stochastic=False)

            @autoflow((tf.float64, [None, None]))
            def predict_stochastic(self, Xnew):
                Xnew = Xnew[None,:,:]
                return self.layer._build_predict(Xnew, stochastic=True)

            @autoflow((tf.float64, [None, None]))
            def predict_full_cov_stochastic(self, Xnew):
                Xnew = Xnew[None,:,:]
                return self.layer._build_predict(Xnew, full_cov=True, stochastic=True)

        return LayerAsModel(layer)

    def test_build_predict_unshared_Z(self):
        X, Z, layer, _, kern_list = self.prepare()
        layer_as_model = self.prepare_autoflow_functions(layer)
        layer_as_model.compile()
        #Variance only and non-stochastic
        with self.subTest():
            m, v = layer_as_model.predict(X)
            self.assertEqual(m.shape, (20, 3))
            self.assertEqual(v.shape, (20, 3))
        with self.subTest():
            m, v = layer_as_model.predict_full_cov(X)
            self.assertEqual(m.shape, (20, 3))
            self.assertEqual(v.shape, (20, 20, 3))
        with self.subTest():
            m, v = layer_as_model.predict_stochastic(X)
            self.assertEqual(m.shape, (1, 20, 3))
            self.assertEqual(v.shape, (1, 20, 3))
        with self.subTest():
            m, v = layer_as_model.predict_full_cov_stochastic(X)
            self.assertEqual(m.shape, (1, 20, 3))
            self.assertEqual(v.shape, (1, 20, 20, 3))


# class InputLayerTest(unittest.TestCase):
    # @defer_build()
    # def setUp(self):
        # self.rng = np.random.RandomState(42)
        # kernel = RBF(2)
        # input_dim = 2
        # output_dim = 2
        # self.W0 = np.zeros((input_dim, output_dim))
        # mean_function = Linear(A=self.W0)
        # self.Z = self.rng.randn(5,2)
        # num_inducing = 5

        # self.layer = InputLayer(input_dim=input_dim,
                                # output_dim=output_dim,
                                # num_inducing=num_inducing,
                                # kernel=kernel,
                                # mean_function=mean_function)

        # self.X = self.rng.randn(10,2)

    # def test_initialize_forward(self):
        # X_running, Z_running = self.layer.initialize_forward(self.X, self.Z)

        # with self.subTest():
            # self.assertFalse(np.allclose(self.layer.mean_function.A.value, self.W0))

        # with self.subTest():
           # self.assertTrue(np.allclose(Z_running, self.Z))

        # with self.subTest():
           # self.assertTrue(np.allclose(self.layer.Z.value, self.Z))

        # with self.subTest():
            # self.assertTrue(np.allclose(X_running, self.X))

# class HiddenLayerTest(unittest.TestCase):
    # @defer_build()
    # def setUp(self):
        # self.rng = np.random.RandomState(42)
        # kernel = RBF(2)
        # input_dim = 2
        # output_dim = 2
        # self.W0 = np.zeros((input_dim, output_dim))
        # mean_function = Linear(A=self.W0)
        # self.Z = self.rng.randn(5, 2)
        # num_inducing = 5

        # self.layer = HiddenLayer(input_dim=input_dim,
                                 # output_dim=output_dim,
                                 # num_inducing=num_inducing,
                                 # kernel=kernel,
                                 # mean_function=mean_function)

        # self.X = self.rng.randn(10, 2)

    # def test_initialize_forward(self):
        # X_running, Z_running = self.layer.initialize_forward(self.X, self.Z)

        # with self.subTest():
            # self.assertFalse(np.allclose(self.layer.mean_function.A.value, self.W0))

        # with self.subTest():
           # self.assertTrue(np.allclose(Z_running, self.Z))

        # with self.subTest():
            # self.assertTrue(np.allclose(X_running, self.X))

        # with self.subTest():
            # self.assertTrue(np.allclose(self.layer.Z.value, self.Z))

# class OutputLayerTest(unittest.TestCase):
    # @defer_build()
    # def setUp(self):
        # self.rng = np.random.RandomState(42)
        # kernel = RBF(2)
        # input_dim = 2
        # output_dim = 2
        # mean_function = None
        # self.Z = self.rng.randn(5, 2)
        # num_inducing = 5

        # self.layer = OutputLayer(input_dim=input_dim,
                                 # output_dim=output_dim,
                                 # num_inducing=num_inducing,
                                 # kernel=kernel,
                                 # mean_function=mean_function)

        # self.X = self.rng.randn(10,2)

    # def test_initialize_forward(self):
        # _ = self.layer.initialize_forward(self.X, self.Z)

        # with self.subTest():
           # self.assertTrue(np.allclose(self.layer.Z.value, self.Z))

if __name__=='__main__':
    unittest.main()
